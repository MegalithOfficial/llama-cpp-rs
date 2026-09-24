//! Safe wrapper around `llama_batch`.

use crate::token::LlamaToken;
use llama_cpp_sys_2::{llama_batch, llama_batch_free, llama_batch_init, llama_pos, llama_seq_id};
use std::ffi::c_void;
use std::marker::PhantomData;

unsafe extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
}

/// A safe wrapper around `llama_batch`.
#[derive(Debug)]
pub struct LlamaBatch<'a> {
    /// The number of tokens the batch was allocated with. they are safe to write to - but not necessarily read from as they are not necessarily initialized
    allocated: usize,
    /// The embedding width allocated for this batch. Zero means token-only.
    n_embd: usize,
    /// Position rows per token: 1, or more for embedding batches of M-RoPE models.
    n_pos_rows: usize,
    /// The logits that are initialized. Used by [`LlamaContext`] to ensure that only initialized logits are accessed.
    pub(crate) initialized_logits: Vec<i32>,
    #[allow(clippy::doc_markdown)]
    /// The llama_cpp batch. always initialize by `llama_cpp_sys_2::llama_batch_init(allocated, <unknown>, <unknown>)`
    pub(crate) llama_batch: llama_batch,
    phantom: PhantomData<&'a [LlamaToken]>,
}

/// Errors that can occur when adding a token to a batch.
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    /// There was not enough space in the batch to add the token.
    #[error("Insufficient Space of {0}")]
    InsufficientSpace(usize),
    /// Empty buffer is provided for [`LlamaBatch::get_one`]
    #[error("Empty buffer")]
    EmptyBuffer,
    /// The batch was created without embedding storage.
    #[error("Batch does not have embedding storage")]
    EmbeddingsDisabled,
    /// The provided embedding row length does not match the batch embedding width.
    #[error("Embedding length mismatch: expected {expected}, got {actual}")]
    EmbeddingLengthMismatch { expected: usize, actual: usize },
    /// The position row is row 0 (written by the `add` methods) or beyond the allocated rows.
    #[error("Position row {row} is not an extra row of a batch with {rows} position rows")]
    PositionRowOutOfRange {
        /// The requested row.
        row: usize,
        /// The rows the batch was allocated with.
        rows: usize,
    },
    /// The position row length does not match the number of tokens in the batch.
    #[error("Position row length mismatch: expected {expected}, got {actual}")]
    PositionRowLengthMismatch {
        /// Tokens in the batch.
        expected: usize,
        /// Positions given.
        actual: usize,
    },
}

impl<'a> LlamaBatch<'a> {
    /// Clear the batch. This does not free the memory associated with the batch, but it does reset
    /// the number of tokens to 0.
    pub fn clear(&mut self) {
        self.llama_batch.n_tokens = 0;
        self.initialized_logits.clear();
    }

    /// add a token to the batch for sequences `seq_ids` at position `pos`. If `logits` is true, the
    /// token will be initialized and can be read from after the next decode.
    ///
    /// # Panics
    ///
    /// - [`self.llama_batch.n_tokens`] does not fit into a usize
    /// - [`seq_ids.len()`] does not fit into a [`llama_seq_id`]
    ///
    /// # Errors
    ///
    /// returns a error if there is insufficient space in the buffer
    pub fn add(
        &mut self,
        LlamaToken(id): LlamaToken,
        pos: llama_pos,
        seq_ids: &[i32],
        logits: bool,
    ) -> Result<(), BatchAddError> {
        if self.allocated
            < usize::try_from(self.n_tokens() + 1).expect("cannot fit n_tokens into a usize")
        {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }
        let offset = self.llama_batch.n_tokens;
        let offset_usize = usize::try_from(offset).expect("cannot fit n_tokens into a usize");
        unsafe {
            // batch.token   [batch.n_tokens] = id; (embeddings-only batches carry no tokens)
            if !self.llama_batch.token.is_null() {
                self.llama_batch.token.add(offset_usize).write(id);
            }
            // batch.pos     [batch.n_tokens] = pos,
            self.llama_batch.pos.add(offset_usize).write(pos);
            // batch.n_seq_id[batch.n_tokens] = seq_ids.size();
            self.llama_batch.n_seq_id.add(offset_usize).write(
                llama_seq_id::try_from(seq_ids.len())
                    .expect("cannot fit seq_ids.len() into a llama_seq_id"),
            );
            // for (size_t i = 0; i < seq_ids.size(); ++i) {
            //     batch.seq_id[batch.n_tokens][i] = seq_ids[i];
            // }
            for (i, seq_id) in seq_ids.iter().enumerate() {
                let tmp = *self.llama_batch.seq_id.add(offset_usize);
                tmp.add(i).write(*seq_id);
            }
            // batch.logits  [batch.n_tokens] = logits;
            self.llama_batch
                .logits
                .add(offset_usize)
                .write(i8::from(logits));
        }

        if logits {
            self.initialized_logits.push(offset);
        } else {
            self.initialized_logits.retain(|l| l != &offset);
        }

        // batch.n_tokens++;
        self.llama_batch.n_tokens += 1;

        Ok(())
    }

    /// Add a token plus embedding row to the batch for MTP-style mixed-token inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if there is insufficient space in the buffer, if the batch was not
    /// created with embedding storage, or if the embedding width does not match.
    pub fn add_with_embedding(
        &mut self,
        token: LlamaToken,
        embedding: &[f32],
        pos: llama_pos,
        seq_ids: &[i32],
        logits: bool,
    ) -> Result<(), BatchAddError> {
        if self.n_embd == 0 || self.llama_batch.embd.is_null() {
            return Err(BatchAddError::EmbeddingsDisabled);
        }
        if embedding.len() != self.n_embd {
            return Err(BatchAddError::EmbeddingLengthMismatch {
                expected: self.n_embd,
                actual: embedding.len(),
            });
        }

        self.add(token, pos, seq_ids, logits)?;

        let offset = usize::try_from(self.llama_batch.n_tokens - 1)
            .expect("cannot fit n_tokens into a usize");
        let embd_offset = offset
            .checked_mul(self.n_embd)
            .expect("embedding offset overflow");

        unsafe {
            std::ptr::copy_nonoverlapping(
                embedding.as_ptr(),
                self.llama_batch.embd.add(embd_offset),
                self.n_embd,
            );
        }

        Ok(())
    }

    /// Add an embedding row without a token, for batches created with
    /// [`Self::new_embeddings_only`].
    ///
    /// # Errors
    ///
    /// Returns an error if there is insufficient space in the buffer, if the batch was not
    /// created with embedding storage, or if the embedding width does not match.
    pub fn add_embedding(
        &mut self,
        embedding: &[f32],
        pos: llama_pos,
        seq_ids: &[i32],
        logits: bool,
    ) -> Result<(), BatchAddError> {
        self.add_with_embedding(LlamaToken(0), embedding, pos, seq_ids, logits)
    }

    /// Add a sequence of tokens to the batch for the given sequence id. If `logits_all` is true, the
    /// tokens will be initialized and can be read from after the next decode.
    ///
    /// Either way the last token in the sequence will have its logits set to `true`.
    ///
    /// # Errors
    ///
    /// Returns an error if there is insufficient space in the buffer
    ///
    /// # Panics
    ///
    /// - [`self.llama_batch.n_tokens`] does not fit into a [`usize`]
    /// - [`n_tokens - 1`] does not fit into a [`llama_pos`]
    pub fn add_sequence(
        &mut self,
        tokens: &[LlamaToken],
        seq_id: i32,
        logits_all: bool,
    ) -> Result<(), BatchAddError> {
        let n_tokens_0 =
            usize::try_from(self.llama_batch.n_tokens).expect("cannot fit n_tokens into a usize");
        let n_tokens = tokens.len();

        if self.allocated < n_tokens_0 + n_tokens {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }

        let last_index = llama_pos::try_from(n_tokens.saturating_sub(1))
            .expect("cannot fit n_tokens into a llama_pos");
        for (i, token) in (0..).zip(tokens.iter()) {
            self.add(*token, i, &[seq_id], logits_all || i == last_index)?;
        }

        Ok(())
    }

    /// Create a new `LlamaBatch` that can contain up to `n_tokens` tokens.
    ///
    /// # Arguments
    ///
    /// - `n_tokens`: the maximum number of tokens that can be added to the batch
    /// - `n_seq_max`: the maximum number of sequences that can be added to the batch (generally 1 unless you know what you are doing)
    ///
    /// # Panics
    ///
    /// Panics if `n_tokens` is greater than `i32::MAX`.
    #[must_use]
    pub fn new(n_tokens: usize, n_seq_max: i32) -> Self {
        let n_tokens_i32 = i32::try_from(n_tokens).expect("cannot fit n_tokens into a i32");
        let batch = unsafe { llama_batch_init(n_tokens_i32, 0, n_seq_max) };

        LlamaBatch {
            allocated: n_tokens,
            n_embd: 0,
            n_pos_rows: 1,
            initialized_logits: vec![],
            llama_batch: batch,
            phantom: PhantomData,
        }
    }

    /// Create a new `LlamaBatch` with storage for both tokens and embedding rows.
    #[must_use]
    pub fn new_with_embeddings(n_tokens: usize, n_embd: usize, n_seq_max: i32) -> Self {
        let n_tokens_i32 = i32::try_from(n_tokens).expect("cannot fit n_tokens into a i32");
        let n_embd_i32 = i32::try_from(n_embd).expect("cannot fit n_embd into a i32");
        let mut batch = unsafe { llama_batch_init(n_tokens_i32, n_embd_i32, n_seq_max) };

        if batch.token.is_null() {
            let bytes = std::mem::size_of::<llama_cpp_sys_2::llama_token>()
                .checked_mul(n_tokens)
                .expect("token allocation overflow");
            let ptr = unsafe { malloc(bytes) } as *mut llama_cpp_sys_2::llama_token;
            assert!(
                !ptr.is_null(),
                "failed to allocate token storage for embedding batch"
            );
            batch.token = ptr;
        }

        LlamaBatch {
            allocated: n_tokens,
            n_embd,
            n_pos_rows: 1,
            initialized_logits: vec![],
            llama_batch: batch,
            phantom: PhantomData,
        }
    }

    /// Create a new `LlamaBatch` that holds only embedding rows: `llama_batch_init` with a
    /// non-zero `n_embd` leaves the token array null, and it stays null, so `llama_decode`
    /// sees a pure embedding batch. Graphs without a token input (such as a `DFlash` drafter's
    /// feature injection) need this; use [`Self::add_embedding`] to fill it.
    ///
    /// # Panics
    ///
    /// Panics if `n_embd` is zero or if `n_tokens` or `n_embd` is greater than `i32::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llama_cpp_2::llama_batch::LlamaBatch;
    /// let mut batch = LlamaBatch::new_embeddings_only(2, 4, 1);
    /// batch.add_embedding(&[0.0; 4], 0, &[0], false).unwrap();
    /// assert_eq!(batch.n_tokens(), 1);
    /// ```
    #[must_use]
    pub fn new_embeddings_only(n_tokens: usize, n_embd: usize, n_seq_max: i32) -> Self {
        Self::new_embeddings_only_with_position_rows(n_tokens, n_embd, n_seq_max, 1)
    }

    /// Like [`Self::new_embeddings_only`], with `n_pos_rows` position rows per token.
    ///
    /// llama.cpp reads an embedding batch of an M-RoPE model as
    /// `pos[row * n_tokens + i]` for each of its position rows (4 for M-RoPE), where a
    /// token batch reuses one row for all of them. The `add` methods write row 0; fill
    /// the other rows with [`Self::set_position_row`] after the last row is added.
    ///
    /// # Panics
    ///
    /// Panics if `n_embd` or `n_pos_rows` is zero, if `n_tokens` or `n_embd` is greater
    /// than `i32::MAX`, or if the position storage cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// # use llama_cpp_2::llama_batch::LlamaBatch;
    /// let mut batch = LlamaBatch::new_embeddings_only_with_position_rows(2, 4, 1, 4);
    /// batch.add_embedding(&[0.0; 4], 7, &[0], false).unwrap();
    /// batch.add_embedding(&[0.0; 4], 8, &[0], false).unwrap();
    /// for row in 1..3 {
    ///     batch.set_position_row(row, &[7, 8]).unwrap();
    /// }
    /// batch.set_position_row(3, &[0, 0]).unwrap();
    /// ```
    #[must_use]
    pub fn new_embeddings_only_with_position_rows(
        n_tokens: usize,
        n_embd: usize,
        n_seq_max: i32,
        n_pos_rows: usize,
    ) -> Self {
        assert!(
            n_embd > 0,
            "an embeddings-only batch needs a non-zero n_embd"
        );
        assert!(n_pos_rows > 0, "a batch needs at least one position row");
        let n_tokens_i32 = i32::try_from(n_tokens).expect("cannot fit n_tokens into a i32");
        let n_embd_i32 = i32::try_from(n_embd).expect("cannot fit n_embd into a i32");
        let mut batch = unsafe { llama_batch_init(n_tokens_i32, n_embd_i32, n_seq_max) };

        if n_pos_rows > 1 {
            let bytes = std::mem::size_of::<llama_pos>()
                .checked_mul(n_tokens)
                .and_then(|bytes| bytes.checked_mul(n_pos_rows))
                .expect("position allocation overflow");
            let ptr = unsafe { malloc(bytes) }.cast::<llama_pos>();
            assert!(!ptr.is_null(), "failed to allocate position rows");
            unsafe { free(batch.pos.cast::<c_void>()) };
            batch.pos = ptr;
        }

        LlamaBatch {
            allocated: n_tokens,
            n_embd,
            n_pos_rows,
            initialized_logits: vec![],
            llama_batch: batch,
            phantom: PhantomData,
        }
    }

    /// Write position row `row` (1 or more; row 0 comes from the `add` methods) for every
    /// token in the batch. Call it after the last token is added: the rows are laid out
    /// by the batch's token count.
    ///
    /// # Errors
    ///
    /// Returns an error if `row` is 0 or not below the batch's position rows, or if
    /// `positions` does not hold one position per token in the batch.
    ///
    /// # Panics
    ///
    /// Panics if the batch's token count does not fit into a `usize`.
    pub fn set_position_row(
        &mut self,
        row: usize,
        positions: &[llama_pos],
    ) -> Result<(), BatchAddError> {
        if row == 0 || row >= self.n_pos_rows {
            return Err(BatchAddError::PositionRowOutOfRange {
                row,
                rows: self.n_pos_rows,
            });
        }
        let n_tokens =
            usize::try_from(self.llama_batch.n_tokens).expect("cannot fit n_tokens into a usize");
        if positions.len() != n_tokens {
            return Err(BatchAddError::PositionRowLengthMismatch {
                expected: n_tokens,
                actual: positions.len(),
            });
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                positions.as_ptr(),
                self.llama_batch.pos.add(row * n_tokens),
                n_tokens,
            );
        }
        Ok(())
    }

    /// ``llama_batch_get_one``
    /// Return batch for single sequence of tokens
    ///
    /// NOTE: this is a helper function to facilitate transition to the new batch API
    ///
    /// # Errors
    /// If the provided token buffer is empty.
    ///
    /// # Panics
    /// If the number of tokens in ``tokens`` exceeds [`i32::MAX`].
    pub fn get_one(tokens: &'a [LlamaToken]) -> Result<Self, BatchAddError> {
        if tokens.is_empty() {
            return Err(BatchAddError::EmptyBuffer);
        }
        let batch = unsafe {
            let ptr = tokens.as_ptr() as *mut i32;
            llama_cpp_sys_2::llama_batch_get_one(
                ptr,
                tokens
                    .len()
                    .try_into()
                    .expect("number of tokens exceeds i32::MAX"),
            )
        };
        let batch = Self {
            allocated: 0,
            n_embd: 0,
            n_pos_rows: 1,
            initialized_logits: vec![(tokens.len() - 1)
                .try_into()
                .expect("number of tokens exceeds i32::MAX + 1")],
            llama_batch: batch,
            phantom: PhantomData,
        };
        Ok(batch)
    }

    /// Returns the number of tokens in the batch.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        self.llama_batch.n_tokens
    }
}

impl<'a> Drop for LlamaBatch<'a> {
    /// Drops the `LlamaBatch`.
    ///
    /// ```
    /// # use llama_cpp_2::llama_batch::LlamaBatch;
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// let batch = LlamaBatch::new(512, 1);
    /// // frees the memory associated with the batch. (allocated by llama.cpp)
    /// drop(batch);
    /// # Ok(())
    /// # }
    fn drop(&mut self) {
        unsafe {
            if self.allocated > 0 {
                llama_batch_free(self.llama_batch);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_only_batches_carry_no_tokens() {
        let mut batch = LlamaBatch::new_embeddings_only(3, 2, 1);
        assert!(batch.llama_batch.token.is_null());
        assert!(!batch.llama_batch.embd.is_null());
        batch.add_embedding(&[1.0, 2.0], 5, &[0], false).unwrap();
        batch
            .add_with_embedding(LlamaToken(9), &[3.0, 4.0], 6, &[0], true)
            .unwrap();
        assert_eq!(batch.n_tokens(), 2);
        assert!(batch.llama_batch.token.is_null());
        let embd = unsafe { std::slice::from_raw_parts(batch.llama_batch.embd, 4) };
        assert_eq!(embd, &[1.0, 2.0, 3.0, 4.0]);
        let pos = unsafe { std::slice::from_raw_parts(batch.llama_batch.pos, 2) };
        assert_eq!(pos, &[5, 6]);
        assert_eq!(batch.initialized_logits, vec![1]);
        assert_eq!(
            batch.add_embedding(&[0.0], 7, &[0], false),
            Err(BatchAddError::EmbeddingLengthMismatch {
                expected: 2,
                actual: 1
            })
        );
        batch.add_embedding(&[5.0, 6.0], 7, &[0], false).unwrap();
        assert_eq!(
            batch.add_embedding(&[0.0, 0.0], 8, &[0], false),
            Err(BatchAddError::InsufficientSpace(3))
        );
    }

    #[test]
    fn extra_position_rows_follow_the_token_count() {
        let mut batch = LlamaBatch::new_embeddings_only_with_position_rows(4, 1, 1, 4);
        assert!(batch.llama_batch.token.is_null());
        batch.add_embedding(&[1.0], 10, &[0], false).unwrap();
        batch.add_embedding(&[2.0], 11, &[0], false).unwrap();
        batch.set_position_row(1, &[10, 11]).unwrap();
        batch.set_position_row(2, &[10, 11]).unwrap();
        batch.set_position_row(3, &[0, 0]).unwrap();
        let pos = unsafe { std::slice::from_raw_parts(batch.llama_batch.pos, 8) };
        assert_eq!(&pos[..8], &[10, 11, 10, 11, 10, 11, 0, 0]);
        assert_eq!(
            batch.set_position_row(0, &[1, 2]),
            Err(BatchAddError::PositionRowOutOfRange { row: 0, rows: 4 })
        );
        assert_eq!(
            batch.set_position_row(4, &[1, 2]),
            Err(BatchAddError::PositionRowOutOfRange { row: 4, rows: 4 })
        );
        assert_eq!(
            batch.set_position_row(1, &[1]),
            Err(BatchAddError::PositionRowLengthMismatch {
                expected: 2,
                actual: 1
            })
        );
        let single = LlamaBatch::new_embeddings_only(1, 1, 1);
        assert_eq!(single.n_pos_rows, 1);
    }

    #[test]
    fn mixed_batches_still_carry_tokens() {
        let mut batch = LlamaBatch::new_with_embeddings(1, 2, 1);
        assert!(!batch.llama_batch.token.is_null());
        batch
            .add_with_embedding(LlamaToken(7), &[1.0, 2.0], 0, &[0], true)
            .unwrap();
        assert_eq!(unsafe { *batch.llama_batch.token }, 7);
    }
}
