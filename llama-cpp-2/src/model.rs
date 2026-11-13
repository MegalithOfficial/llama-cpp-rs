//! A safe wrapper around `llama_model`.
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::slice;
use std::str::Utf8Error;

use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::model::params::LlamaModelParams;
use crate::ptr::Ptr;
#[cfg(feature = "common")]
use crate::openai::{ChatParseStateOaicompat, OpenAIChatTemplateParams};
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;
use crate::vocab::LlamaVocab;
#[cfg(feature = "common")]
use crate::{status_is_ok, ChatParseError};
use crate::{
    ApplyChatTemplateError, ChatTemplateError, LlamaContextLoadError, LlamaLoraAdapterInitError,
    LlamaModelLoadError, MetaValError, NewLlamaChatMessageError,
};
#[cfg(feature = "common")]
use crate::KvCacheTypeError;

pub mod params;
// For backwards compat.
pub use crate::vocab::{LlamaTokenTypeFromIntError, VocabType};

/// A safe wrapper around `llama_model`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaModel {
    pub(crate) model: Ptr<llama_cpp_sys_2::llama_model>,
}

// SAFETY: The model is immutable except for methods where it's passed as `&mut`.
//
// FIXME(madsmtm): Apart from `llama_new_context_with_model`, which mutates.
unsafe impl Send for LlamaModel {}
// SAFETY: Same as above.
unsafe impl Sync for LlamaModel {}

/// A safe wrapper around `llama_lora_adapter`.
#[derive(Debug)]
#[repr(transparent)]
#[allow(clippy::module_name_repetitions)]
pub struct LlamaLoraAdapter {
    pub(crate) lora_adapter: Ptr<llama_cpp_sys_2::llama_adapter_lora>,
}

// SAFETY: The lora is immutable.
unsafe impl Send for LlamaLoraAdapter {}
// SAFETY: Same as above.
unsafe impl Sync for LlamaLoraAdapter {}

/// A performance-friendly wrapper around [`LlamaModel::chat_template`] which is then
/// fed into [`LlamaModel::apply_chat_template`] to convert a list of messages into an LLM
/// prompt. Internally the template is stored as a `CString` to avoid round-trip conversions
/// within the FFI.
#[derive(Eq, PartialEq, Clone, PartialOrd, Ord, Hash)]
pub struct LlamaChatTemplate(CString);

impl LlamaChatTemplate {
    /// Create a new template from a string. This can either be the name of a llama.cpp [chat template](https://github.com/ggerganov/llama.cpp/blob/8a8c4ceb6050bd9392609114ca56ae6d26f5b8f5/src/llama-chat.cpp#L27-L61)
    /// like "chatml" or "llama3" or an actual Jinja template for llama.cpp to interpret.
    pub fn new(template: &str) -> Result<Self, std::ffi::NulError> {
        Ok(Self(CString::new(template)?))
    }

    /// Accesses the template as a c string reference.
    pub fn as_c_str(&self) -> &CStr {
        &self.0
    }

    /// Attempts to convert the `CString` into a Rust str reference.
    pub fn to_str(&self) -> Result<&str, Utf8Error> {
        self.0.to_str()
    }

    /// Convenience method to create an owned String.
    pub fn to_string(&self) -> Result<String, Utf8Error> {
        self.to_str().map(str::to_string)
    }
}

impl std::fmt::Debug for LlamaChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A Safe wrapper around `llama_chat_message`
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    role: CString,
    content: CString,
}

impl LlamaChatMessage {
    /// Create a new `LlamaChatMessage`
    ///
    /// # Errors
    /// If either of ``role`` or ``content`` contain null bytes.
    pub fn new(role: String, content: String) -> Result<Self, NewLlamaChatMessageError> {
        Ok(Self {
            role: CString::new(role)?,
            content: CString::new(content)?,
        })
    }
}

/// Grammar trigger kinds used for lazy grammar sampling.
#[cfg(feature = "common")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarTriggerType {
    /// Trigger on a specific token.
    Token = 0,
    /// Trigger on a literal word.
    Word = 1,
    /// Trigger on a regex pattern.
    Pattern = 2,
    /// Trigger on a fully anchored regex pattern.
    PatternFull = 3,
}

/// Lazy grammar trigger from chat template generation.
#[cfg(feature = "common")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarTrigger {
    /// Trigger kind.
    pub trigger_type: GrammarTriggerType,
    /// Trigger text or pattern.
    pub value: String,
    /// Token id for token triggers.
    pub token: Option<LlamaToken>,
}

/// PEG parser built by a chat template for parsing its responses.
#[cfg(feature = "common")]
#[derive(Clone)]
pub struct ChatParser(std::sync::Arc<ChatParserHandle>);

#[cfg(feature = "common")]
struct ChatParserHandle(NonNull<llama_cpp_sys_2::llama_rs_chat_parser>);

// SAFETY: The parser is immutable after construction and only read by llama.cpp.
#[cfg(feature = "common")]
unsafe impl Send for ChatParserHandle {}
// SAFETY: Same as above.
#[cfg(feature = "common")]
unsafe impl Sync for ChatParserHandle {}

#[cfg(feature = "common")]
impl Drop for ChatParserHandle {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_rs_chat_parser_free(self.0.as_ptr()) }
    }
}

#[cfg(feature = "common")]
impl ChatParser {
    fn as_ptr(&self) -> *const llama_cpp_sys_2::llama_rs_chat_parser {
        self.0 .0.as_ptr().cast_const()
    }
}

#[cfg(feature = "common")]
impl std::fmt::Debug for ChatParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ChatParser").field(&self.0 .0).finish()
    }
}

#[cfg(feature = "common")]
impl PartialEq for ChatParser {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

#[cfg(feature = "common")]
impl Eq for ChatParser {}

/// Result of applying a chat template with tool grammar support.
#[cfg(feature = "common")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTemplateResult {
    /// Rendered chat prompt.
    pub prompt: String,
    /// Optional grammar generated from tool definitions.
    pub grammar: Option<String>,
    /// Whether to use lazy grammar sampling.
    pub grammar_lazy: bool,
    /// Lazy grammar triggers derived from the template.
    pub grammar_triggers: Vec<GrammarTrigger>,
    /// Tokens that should be preserved for sampling.
    pub preserved_tokens: Vec<String>,
    /// Additional stop sequences added by the template.
    pub additional_stops: Vec<String>,
    /// Chat format used for parsing responses.
    pub chat_format: i32,
    /// Optional PEG parser for tool-call parsing.
    pub parser: Option<ChatParser>,
    /// Prefix that must be prepended for parser-compatible response reconstruction.
    pub generation_prompt: String,
    /// Whether tool calls should be parsed from the response.
    pub parse_tool_calls: bool,
    pub supports_thinking: bool,
    pub thinking_start_tag: Option<String>,
    pub thinking_end_tags: Vec<String>,
}

#[cfg(feature = "common")]
impl ChatTemplateResult {
    #[must_use]
    pub fn thinking_forced_open(&self) -> bool {
        let Some(start) = self.thinking_start_tag.as_deref().filter(|s| !s.is_empty()) else {
            return false;
        };
        let Some(open_at) = self.generation_prompt.rfind(start) else {
            return false;
        };
        let after_open = &self.generation_prompt[open_at + start.len()..];
        !self
            .thinking_end_tags
            .iter()
            .any(|end| !end.is_empty() && after_open.contains(end.as_str()))
    }
}

#[cfg(feature = "common")]
unsafe fn thinking_metadata_from_raw(
    raw: &llama_cpp_sys_2::llama_rs_chat_template_result,
) -> Result<(Option<String>, Vec<String>), ApplyChatTemplateError> {
    let start_tag = if raw.thinking_start_tag.is_null() {
        None
    } else {
        let bytes = unsafe { CStr::from_ptr(raw.thinking_start_tag) }
            .to_bytes()
            .to_vec();
        Some(String::from_utf8(bytes)?)
    };

    let end_tags = if raw.thinking_end_tags_count == 0 || raw.thinking_end_tags.is_null() {
        Vec::new()
    } else {
        let tags =
            unsafe { slice::from_raw_parts(raw.thinking_end_tags, raw.thinking_end_tags_count) };
        let mut parsed = Vec::with_capacity(tags.len());
        for tag in tags {
            if tag.is_null() {
                continue;
            }
            let bytes = unsafe { CStr::from_ptr(*tag) }.to_bytes().to_vec();
            parsed.push(String::from_utf8(bytes)?);
        }
        parsed
    };

    Ok((start_tag, end_tags))
}

#[cfg(feature = "common")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayerGeometry {
    pub n_head_kv: u32,
    pub n_embd_head_k: u32,
    pub n_embd_head_v: u32,
    pub is_swa: bool,
}

#[cfg(feature = "common")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvGeometry {
    pub layers: Vec<KvLayerGeometry>,
    pub n_swa: u32,
}

/// The RoPE type that's used within the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// Normal RoPE.
    Norm,
    /// Neox-style dimension interleave.
    NeoX,
    /// multi-section rotary embedding (MRoPE).
    MRope,
    /// Interleaved MRoPE.
    IMRope,
    /// Vision RoPE.
    Vision,
}

impl LlamaModel {
    /// Get the model's vocabulary.
    #[must_use]
    pub fn vocab(&self) -> LlamaVocab<'_> {
        let ptr = unsafe { llama_cpp_sys_2::llama_model_get_vocab(self.model.as_ptr()) };
        LlamaVocab::new(ptr).expect("model must have vocabulary")
    }

    /// get the number of tokens the model was trained on
    ///
    /// # Panics
    ///
    /// If the number of tokens the model was trained on does not fit into an `u32`. This should be impossible on most
    /// platforms due to llama.cpp returning a `c_int` (i32 on most platforms) which is almost certainly positive.
    #[must_use]
    pub fn n_ctx_train(&self) -> u32 {
        let n_ctx_train = unsafe { llama_cpp_sys_2::llama_model_n_ctx_train(self.model.as_ptr()) };
        u32::try_from(n_ctx_train).expect("n_ctx_train fits into an u32")
    }

    /// Get all tokens in the model.
    pub fn tokens(&self, decode_special: bool) -> impl Iterator<Item = (LlamaToken, Vec<u8>)> + '_ {
        let vocab = self.vocab();
        vocab.tokens().map(move |llama_token| {
            let bytes = vocab.token_to_piece(llama_token, decode_special, None);
            (llama_token, bytes)
        })
    }

    /// Get the decoder start token.
    #[must_use]
    pub fn decode_start_token(&self) -> LlamaToken {
        let token =
            unsafe { llama_cpp_sys_2::llama_model_decoder_start_token(self.model.as_ptr()) };
        LlamaToken(token)
    }

    /// The number of tokens the model was trained on.
    ///
    /// This returns a `c_int` for maximum compatibility. Most of the time it can be cast to an i32
    /// without issue.
    #[must_use]
    pub fn n_vocab(&self) -> i32 {
        self.vocab().n_tokens()
    }

    /// This returns a `c_int` for maximum compatibility. Most of the time it can be cast to an i32
    /// without issue.
    #[must_use]
    pub fn n_embd(&self) -> c_int {
        unsafe { llama_cpp_sys_2::llama_model_n_embd(self.model.as_ptr()) }
    }

    /// Get the input embedding width of the model.
    #[must_use]
    pub fn n_embd_inp(&self) -> c_int {
        unsafe { llama_cpp_sys_2::llama_model_n_embd_inp(self.model.as_ptr()) }
    }

    /// The model's *output* embedding width (`n_embd_out`). This is the width
    /// llama.cpp actually extracts embeddings at — `n_embd` and `n_embd_out`
    /// diverge when `{arch}.embedding_length_out` is present (deepstack models
    /// like qwen3vl). Returns a `c_int` for maximum compatibility.
    #[must_use]
    pub fn n_embd_out(&self) -> c_int {
        unsafe { llama_cpp_sys_2::llama_model_n_embd_out(self.model.as_ptr()) }
    }

    /// Check the context's K/V cache types against what llama.cpp will accept for this model.
    ///
    /// `new_context` reports these as [`LlamaContextLoadError::KvCacheType`], so calling this
    /// directly is only needed to reject a configuration before the model is even loaded.
    ///
    /// # Errors
    ///
    /// See [`KvCacheTypeError`].
    #[cfg(feature = "common")]
    pub fn validate_kv_cache_types(
        &self,
        params: &LlamaContextParams,
    ) -> Result<(), KvCacheTypeError> {
        let type_k = params.type_k();
        let type_v = params.type_v();
        let fa_disabled = params.flash_attention_policy()
            == llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED;

        if type_v.is_quantized() && fa_disabled {
            return Err(KvCacheTypeError::QuantizedVNeedsFlashAttention);
        }
        if type_k != type_v
            && unsafe {
                llama_cpp_sys_2::llama_rs_model_requires_uniform_kv_type(self.model.as_ptr())
            }
        {
            return Err(KvCacheTypeError::MixedTypesUnsupported);
        }
        if fa_disabled {
            return Ok(());
        }

        let Some(geometry) = self.kv_geometry() else {
            return Ok(());
        };
        for (layer, dims) in geometry.layers.iter().enumerate() {
            if type_k.is_quantized() {
                let block_size = type_k.block_size();
                if dims.n_embd_head_k % block_size != 0 {
                    return Err(KvCacheTypeError::KBlockSize {
                        layer,
                        block_size,
                        n_embd_head: dims.n_embd_head_k,
                    });
                }
            }
            if type_v.is_quantized() {
                let block_size = type_v.block_size();
                if dims.n_embd_head_v % block_size != 0 {
                    return Err(KvCacheTypeError::VBlockSize {
                        layer,
                        block_size,
                        n_embd_head: dims.n_embd_head_v,
                    });
                }
            }
        }

        Ok(())
    }

    /// Whether `draft` can draft for this model, by llama.cpp's speculative vocabulary
    /// check (`common_speculative_are_compatible`): the same vocab type, the same BOS and
    /// EOS use and ids, vocab sizes at most 128 apart, and the same text for every token
    /// id from 5 up to the smaller vocab.
    #[must_use]
    pub fn is_speculation_compatible(&self, draft: &LlamaModel) -> bool {
        const MAX_SIZE_DIFFERENCE: i32 = 128;
        const CHECK_START_TOKEN_ID: i32 = 5;
        let target_vocab = self.vocab();
        let draft_vocab = draft.vocab();
        let target = target_vocab.as_ptr();
        let draft = draft_vocab.as_ptr();
        unsafe {
            if llama_cpp_sys_2::llama_vocab_type(target) != llama_cpp_sys_2::llama_vocab_type(draft)
            {
                return false;
            }
            let bos_added = llama_cpp_sys_2::llama_vocab_get_add_bos(target);
            if bos_added != llama_cpp_sys_2::llama_vocab_get_add_bos(draft)
                || (bos_added
                    && llama_cpp_sys_2::llama_vocab_bos(target)
                        != llama_cpp_sys_2::llama_vocab_bos(draft))
            {
                return false;
            }
            let eos_added = llama_cpp_sys_2::llama_vocab_get_add_eos(target);
            if eos_added != llama_cpp_sys_2::llama_vocab_get_add_eos(draft)
                || (eos_added
                    && llama_cpp_sys_2::llama_vocab_eos(target)
                        != llama_cpp_sys_2::llama_vocab_eos(draft))
            {
                return false;
            }
            let n_target = llama_cpp_sys_2::llama_vocab_n_tokens(target);
            let n_draft = llama_cpp_sys_2::llama_vocab_n_tokens(draft);
            if (n_target - n_draft).abs() > MAX_SIZE_DIFFERENCE {
                return false;
            }
            for id in CHECK_START_TOKEN_ID..n_target.min(n_draft) {
                let target_text = llama_cpp_sys_2::llama_vocab_get_text(target, id);
                let draft_text = llama_cpp_sys_2::llama_vocab_get_text(draft, id);
                if target_text.is_null() || draft_text.is_null() {
                    if target_text != draft_text {
                        return false;
                    }
                    continue;
                }
                if CStr::from_ptr(target_text) != CStr::from_ptr(draft_text) {
                    return false;
                }
            }
        }
        true
    }

    #[must_use]
    pub fn dflash_selector_top_k(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_rs_model_dflash_selector_top_k(self.model.as_ptr()) }
    }

    #[must_use]
    pub fn target_layer_ids(&self) -> &[i32] {
        unsafe {
            let ids = llama_cpp_sys_2::llama_rs_model_target_layer_ids(self.model.as_ptr());
            let n = llama_cpp_sys_2::llama_rs_model_target_layer_ids_n(self.model.as_ptr());
            if ids.is_null() || n == 0 {
                &[]
            } else {
                slice::from_raw_parts(ids, n as usize)
            }
        }
    }

    /// The model's classification output width (`n_cls_out`, default 1) — the
    /// width of a RANK-pooled embeddings read (llama.h:1029).
    #[must_use]
    pub fn n_cls_out(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_model_n_cls_out(self.model.as_ptr()) }
    }
    /// Returns the total size of all the tensors in the model in bytes.
    pub fn size(&self) -> u64 {
        unsafe { llama_cpp_sys_2::llama_model_size(self.model.as_ptr()) }
    }

    /// Returns the number of parameters in the model.
    pub fn n_params(&self) -> u64 {
        unsafe { llama_cpp_sys_2::llama_model_n_params(self.model.as_ptr()) }
    }

    /// Returns whether the model is a recurrent network (Mamba, RWKV, etc)
    pub fn is_recurrent(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_model_is_recurrent(self.model.as_ptr()) }
    }

    /// Returns whether the model is a hybrid network (Jamba, Granite, Qwen3xx, etc)
    ///
    /// Hybrid models have both attention layers and recurrent/SSM layers.
    /// They require special handling for state checkpointing.
    pub fn is_hybrid(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_model_is_hybrid(self.model.as_ptr()) }
    }

    /// Returns the number of layers within the model.
    pub fn n_layer(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_layer(self.model.as_ptr()) }).unwrap()
    }

    /// Returns the number of next-token-prediction layers within the model.
    pub fn n_layer_nextn(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_layer_nextn(self.model.as_ptr()) })
            .unwrap()
    }

    /// Returns the number of attention heads within the model.
    pub fn n_head(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_head(self.model.as_ptr()) }).unwrap()
    }

    /// Returns the number of KV attention heads.
    pub fn n_head_kv(&self) -> u32 {
        // It's never possible for this to panic because while the API interface is defined as an int32_t,
        // the field it's accessing is a uint32_t.
        u32::try_from(unsafe { llama_cpp_sys_2::llama_model_n_head_kv(self.model.as_ptr()) })
            .unwrap()
    }

    #[cfg(feature = "common")]
    #[must_use]
    pub fn kv_geometry(&self) -> Option<KvGeometry> {
        let mut raw: *mut llama_cpp_sys_2::llama_rs_kv_layer_geometry = ptr::null_mut();
        let mut count: usize = 0;
        let mut n_swa: u32 = 0;
        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_model_kv_geometry(
                self.model.as_ptr(),
                &mut raw,
                &mut count,
                &mut n_swa,
            )
        };
        if !status_is_ok(rc) || raw.is_null() || count == 0 {
            if !raw.is_null() {
                unsafe { llama_cpp_sys_2::llama_rs_kv_geometry_free(raw) };
            }
            return None;
        }
        let layers = unsafe { slice::from_raw_parts(raw, count) }
            .iter()
            .map(|layer| KvLayerGeometry {
                n_head_kv: layer.n_head_kv,
                n_embd_head_k: layer.n_embd_head_k,
                n_embd_head_v: layer.n_embd_head_v,
                is_swa: layer.is_swa,
            })
            .collect();
        unsafe { llama_cpp_sys_2::llama_rs_kv_geometry_free(raw) };
        Some(KvGeometry { layers, n_swa })
    }

    /// Get metadata value as a string by key name
    pub fn meta_val_str(&self, key: &str) -> Result<String, MetaValError> {
        let key_cstring = CString::new(key)?;
        let key_ptr = key_cstring.as_ptr();

        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_val_str(
                    self.model.as_ptr(),
                    key_ptr,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Get the number of metadata key/value pairs
    pub fn meta_count(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_model_meta_count(self.model.as_ptr()) }
    }

    /// Get metadata key name by index
    pub fn meta_key_by_index(&self, index: i32) -> Result<String, MetaValError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_key_by_index(
                    self.model.as_ptr(),
                    index,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Get metadata value as a string by index
    pub fn meta_val_str_by_index(&self, index: i32) -> Result<String, MetaValError> {
        extract_meta_string(
            |buf_ptr, buf_len| unsafe {
                llama_cpp_sys_2::llama_model_meta_val_str_by_index(
                    self.model.as_ptr(),
                    index,
                    buf_ptr,
                    buf_len,
                )
            },
            256,
        )
    }

    /// Returns the rope type of the model.
    pub fn rope_type(&self) -> Option<RopeType> {
        match unsafe { llama_cpp_sys_2::llama_model_rope_type(self.model.as_ptr()) } {
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NONE => None,
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NORM => Some(RopeType::Norm),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_NEOX => Some(RopeType::NeoX),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_MROPE => Some(RopeType::MRope),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_IMROPE => Some(RopeType::IMRope),
            llama_cpp_sys_2::LLAMA_ROPE_TYPE_VISION => Some(RopeType::Vision),
            rope_type => {
                tracing::error!(rope_type = rope_type, "Unexpected rope type from llama.cpp");
                None
            }
        }
    }

    /// Get chat template from model by name. If the name parameter is None, the default chat template will be returned.
    ///
    /// You supply this into [`Self::apply_chat_template`] to get back a string with the appropriate template
    /// substitution applied to convert a list of messages into a prompt the LLM can use to complete
    /// the chat.
    ///
    /// You could also use an external jinja parser, like [minijinja](https://github.com/mitsuhiko/minijinja),
    /// to parse jinja templates not supported by the llama.cpp template engine.
    ///
    /// # Errors
    ///
    /// * If the model has no chat template by that name
    /// * If the chat template is not a valid [`CString`].
    pub fn chat_template(
        &self,
        name: Option<&str>,
    ) -> Result<LlamaChatTemplate, ChatTemplateError> {
        let name_cstr = name.map(CString::new);
        let name_ptr = match name_cstr {
            Some(Ok(name)) => name.as_ptr(),
            _ => std::ptr::null(),
        };
        let result =
            unsafe { llama_cpp_sys_2::llama_model_chat_template(self.model.as_ptr(), name_ptr) };

        // Convert result to Rust String if not null
        if result.is_null() {
            Err(ChatTemplateError::MissingTemplate)
        } else {
            let chat_template_cstr = unsafe { CStr::from_ptr(result) };
            let chat_template = CString::new(chat_template_cstr.to_bytes())?;
            Ok(LlamaChatTemplate(chat_template))
        }
    }

    /// Loads a model from a file.
    ///
    /// # Errors
    ///
    /// See [`LlamaModelLoadError`] for more information.
    #[tracing::instrument(skip_all, fields(params))]
    pub fn load_from_file(
        _: &LlamaBackend,
        path: impl AsRef<Path>,
        params: &LlamaModelParams,
    ) -> Result<Self, LlamaModelLoadError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");
        let path = path
            .to_str()
            .ok_or(LlamaModelLoadError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let llama_model =
            unsafe { llama_cpp_sys_2::llama_model_load_from_file(cstr.as_ptr(), params.params) };

        let model = Ptr::new(llama_model).ok_or(LlamaModelLoadError::NullResult)?;

        tracing::debug!(?path, "Loaded model");
        Ok(LlamaModel { model })
    }

    /// Initializes a lora adapter from a file.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterInitError`] for more information.
    pub fn lora_adapter_init(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<LlamaLoraAdapter, LlamaLoraAdapterInitError> {
        let path = path.as_ref();
        debug_assert!(Path::new(path).exists(), "{path:?} does not exist");

        let path = path
            .to_str()
            .ok_or(LlamaLoraAdapterInitError::PathToStrError(
                path.to_path_buf(),
            ))?;

        let cstr = CString::new(path)?;
        let adapter = unsafe {
            llama_cpp_sys_2::llama_adapter_lora_init(self.model.as_mut_ptr(), cstr.as_ptr())
        };

        let adapter = Ptr::new(adapter).ok_or(LlamaLoraAdapterInitError::NullResult)?;

        tracing::debug!(?path, "Initialized lora adapter");
        Ok(LlamaLoraAdapter {
            lora_adapter: adapter,
        })
    }

    /// Create a new context from this model.
    ///
    /// # Errors
    ///
    /// There is many ways this can fail. See [`LlamaContextLoadError`] for more information.
    // we intentionally do not derive Copy on `LlamaContextParams` to allow llama.cpp to change the type to be non-trivially copyable.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context<'a>(
        &'a self,
        _: &LlamaBackend,
        params: LlamaContextParams,
    ) -> Result<LlamaContext<'a>, LlamaContextLoadError> {
        let context_params = params.context_params;
        // Constructing multiple contexts from a model in parallel is actually
        // unsound, since `llama_new_context_with_model` mutates the model's
        // `n_ctx_train` in some cases:
        // <https://github.com/ggml-org/llama.cpp/blob/f45576aa86c07d60d346b469651a098a15cc4f81/src/llama-context.cpp#L3751>
        // FIXME(madsmtm): Make this sound!
        let context = unsafe {
            llama_cpp_sys_2::llama_init_from_model(
                self.model.as_mut_ptr_unsound(),
                context_params,
            )
        };
        let context = Ptr::new(context).ok_or_else(|| self.context_load_error(&params))?;

        Ok(LlamaContext::new(context))
    }

    /// Explain a null return from `llama_new_context_with_model` where we can.
    fn context_load_error(&self, params: &LlamaContextParams) -> LlamaContextLoadError {
        #[cfg(feature = "common")]
        if let Err(err) = self.validate_kv_cache_types(params) {
            return LlamaContextLoadError::KvCacheType(err);
        }
        let _ = params;
        LlamaContextLoadError::NullReturn
    }

    /// Create a new context bound to another context via llama.cpp's `ctx_other` field.
    ///
    /// This is required for MTP speculative decoding when the target model's
    /// architecture uses `LLM_ARCH_GEMMA4_ASSISTANT`, which asserts that the draft
    /// context references the target context so KV state can be shared.
    ///
    /// # Errors
    ///
    /// See [`LlamaContextLoadError`].
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context_with_ctx_other<'a>(
        &'a self,
        _: &LlamaBackend,
        params: LlamaContextParams,
        ctx_other: &LlamaContext<'_>,
    ) -> Result<LlamaContext<'a>, LlamaContextLoadError> {
        let mut context_params = params.context_params;
        // FIXME(madsmtm): Use `.as_ptr()` after:
        // https://github.com/ggml-org/llama.cpp/pull/28316
        context_params.ctx_other = unsafe { ctx_other.context.as_mut_ptr_unsound() };
        // Unsoundness: See `LlamaMode::new_context`.
        let context = unsafe {
            llama_cpp_sys_2::llama_init_from_model(
                self.model.as_mut_ptr_unsound(),
                context_params,
            )
        };
        let context = Ptr::new(context).ok_or_else(|| self.context_load_error(&params))?;

        Ok(LlamaContext::new(context))
    }

    /// Creates a new context with backend samplers attached for specific sequences.
    ///
    /// Ownership of the samplers is transferred to the context, ensuring they remain
    /// alive for the context's lifetime. Only samplers that support backend execution
    /// (greedy, dist, temp, top_k, top_p, min_p, logit_bias) will run on the backend.
    ///
    /// # Arguments
    ///
    /// * `params` - Context parameters
    /// * `samplers` - Iterator of `(seq_id, sampler)` pairs where sampler must be a chain
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sampler = LlamaSampler::chain([
    ///     LlamaSampler::min_p(0.01, 64),
    ///     LlamaSampler::temp(0.1),
    ///     LlamaSampler::dist(42),
    /// ], false);
    ///
    /// let ctx = model.new_context_with_samplers(
    ///     &backend,
    ///     ctx_params,
    ///     [(0, sampler)],
    /// )?;
    /// ```
    #[allow(clippy::needless_pass_by_value)]
    pub fn new_context_with_samplers<'a>(
        &'a self,
        _: &LlamaBackend,
        params: LlamaContextParams,
        samplers: impl IntoIterator<Item = (i32, LlamaSampler)>,
    ) -> Result<LlamaContext<'a>, LlamaContextLoadError> {
        let mut samplers: Vec<_> = samplers.into_iter().collect();
        let mut context_params = params.context_params;

        let mut sampler_configs: Vec<llama_cpp_sys_2::llama_sampler_seq_config> = samplers
            .iter_mut()
            .map(
                |(seq_id, sampler)| llama_cpp_sys_2::llama_sampler_seq_config {
                    seq_id: *seq_id,
                    sampler: sampler.sampler.as_mut_ptr(),
                },
            )
            .collect();

        if !sampler_configs.is_empty() {
            context_params.samplers = sampler_configs.as_mut_ptr();
            context_params.n_samplers = sampler_configs.len();
        }

        // Unsoundness: See `LlamaMode::new_context`.
        let context = unsafe {
            llama_cpp_sys_2::llama_init_from_model(
                self.model.as_mut_ptr_unsound(),
                context_params,
            )
        };
        let context = Ptr::new(context).ok_or_else(|| self.context_load_error(&params))?;

        Ok(LlamaContext::with_samplers(context, samplers))
    }

    /// Apply the models chat template to some messages.
    /// See <https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template>
    ///
    /// Unlike the llama.cpp `apply_chat_template` which just randomly uses the ChatML template when given
    /// a null pointer for the template, this requires an explicit template to be specified. If you want to
    /// use "chatml", then just do `LlamaChatTemplate::new("chatml")` or any other model name or template
    /// string.
    ///
    /// Use [`Self::chat_template`] to retrieve the template baked into the model (this is the preferred
    /// mechanism as using the wrong chat template can result in really unexpected responses from the LLM).
    ///
    /// You probably want to set `add_ass` to true so that the generated template string ends with a the
    /// opening tag of the assistant. If you fail to leave a hanging chat tag, the model will likely generate
    /// one into the output and the output may also have unexpected output aside from that.
    ///
    /// # Errors
    /// There are many ways this can fail. See [`ApplyChatTemplateError`] for more information.
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template(
        &self,
        tmpl: &LlamaChatTemplate,
        chat: &[LlamaChatMessage],
        add_ass: bool,
    ) -> Result<String, ApplyChatTemplateError> {
        // Buffer is twice the length of messages per their recommendation
        let message_length = chat.iter().fold(0, |acc, c| {
            acc + c.role.to_bytes().len() + c.content.to_bytes().len()
        });
        let mut buff: Vec<u8> = vec![0; message_length * 2];

        // Build our llama_cpp_sys_2 chat messages
        let chat: Vec<llama_cpp_sys_2::llama_chat_message> = chat
            .iter()
            .map(|c| llama_cpp_sys_2::llama_chat_message {
                role: c.role.as_ptr(),
                content: c.content.as_ptr(),
            })
            .collect();

        let tmpl_ptr = tmpl.0.as_ptr();

        let res = unsafe {
            llama_cpp_sys_2::llama_chat_apply_template(
                tmpl_ptr,
                chat.as_ptr(),
                chat.len(),
                add_ass,
                buff.as_mut_ptr().cast::<c_char>(),
                buff.len().try_into().expect("Buffer size exceeds i32::MAX"),
            )
        };

        if res < 0 {
            return Err(ApplyChatTemplateError::FfiError(res));
        }

        if res > buff.len().try_into().expect("Buffer size exceeds i32::MAX") {
            buff.resize(res.try_into().expect("res is negative"), 0);

            let res = unsafe {
                llama_cpp_sys_2::llama_chat_apply_template(
                    tmpl_ptr,
                    chat.as_ptr(),
                    chat.len(),
                    add_ass,
                    buff.as_mut_ptr().cast::<c_char>(),
                    buff.len().try_into().expect("Buffer size exceeds i32::MAX"),
                )
            };
            if res < 0 {
                return Err(ApplyChatTemplateError::FfiError(res));
            }
            assert_eq!(Ok(res), buff.len().try_into());
        }
        buff.truncate(res.try_into().expect("res is negative"));
        Ok(String::from_utf8(buff)?)
    }

    /// Apply the models chat template to some messages and return an optional tool grammar.
    /// `tools_json` should be an OpenAI-compatible tool definition JSON array string.
    /// `json_schema` should be a JSON schema string.
    #[cfg(feature = "common")]
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template_with_tools_oaicompat(
        &self,
        tmpl: &LlamaChatTemplate,
        messages: &[LlamaChatMessage],
        tools_json: Option<&str>,
        json_schema: Option<&str>,
        add_generation_prompt: bool,
    ) -> Result<ChatTemplateResult, ApplyChatTemplateError> {
        let chat: Vec<llama_cpp_sys_2::llama_chat_message> = messages
            .iter()
            .map(|c| llama_cpp_sys_2::llama_chat_message {
                role: c.role.as_ptr(),
                content: c.content.as_ptr(),
            })
            .collect();

        let tools_cstr = tools_json.map(CString::new).transpose()?;
        let json_schema_cstr = json_schema.map(CString::new).transpose()?;

        let mut raw_result = llama_cpp_sys_2::llama_rs_chat_template_result {
            prompt: ptr::null_mut(),
            grammar: ptr::null_mut(),
            parser: ptr::null_mut(),
            generation_prompt: ptr::null_mut(),
            chat_format: 0,
            grammar_lazy: false,
            grammar_triggers: ptr::null_mut(),
            grammar_triggers_count: 0,
            preserved_tokens: ptr::null_mut(),
            preserved_tokens_count: 0,
            additional_stops: ptr::null_mut(),
            additional_stops_count: 0,
            supports_thinking: false,
            thinking_start_tag: ptr::null_mut(),
            thinking_end_tags: ptr::null_mut(),
            thinking_end_tags_count: 0,
        };

        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_apply_chat_template_with_tools_oaicompat(
                self.model.as_ptr(),
                tmpl.0.as_ptr(),
                chat.as_ptr(),
                chat.len(),
                tools_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                json_schema_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                add_generation_prompt,
                &mut raw_result,
            )
        };

        let result = (|| {
            if !status_is_ok(rc) {
                return Err(match crate::status_error_message(rc) {
                    Some(message) => ApplyChatTemplateError::FfiException(rc, message),
                    None => ApplyChatTemplateError::FfiError(rc),
                });
            }
            if raw_result.prompt.is_null() {
                return Err(ApplyChatTemplateError::NullResult);
            }
            let prompt_bytes = unsafe { CStr::from_ptr(raw_result.prompt) }
                .to_bytes()
                .to_vec();
            let prompt = String::from_utf8(prompt_bytes)?;
            let grammar_lazy = raw_result.grammar_lazy;
            let grammar = if raw_result.grammar.is_null() {
                None
            } else {
                let grammar_bytes = unsafe { CStr::from_ptr(raw_result.grammar) }
                    .to_bytes()
                    .to_vec();
                Some(String::from_utf8(grammar_bytes)?)
            };
            let parser = NonNull::new(std::mem::replace(&mut raw_result.parser, ptr::null_mut()))
                .map(|handle| ChatParser(std::sync::Arc::new(ChatParserHandle(handle))));
            let generation_prompt = if raw_result.generation_prompt.is_null() {
                String::new()
            } else {
                let generation_prompt_bytes =
                    unsafe { CStr::from_ptr(raw_result.generation_prompt) }
                        .to_bytes()
                        .to_vec();
                String::from_utf8(generation_prompt_bytes)?
            };
            let grammar_triggers = if raw_result.grammar_triggers_count == 0 {
                Vec::new()
            } else if raw_result.grammar_triggers.is_null() {
                return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
            } else {
                let triggers = unsafe {
                    slice::from_raw_parts(
                        raw_result.grammar_triggers,
                        raw_result.grammar_triggers_count,
                    )
                };
                let mut parsed = Vec::with_capacity(triggers.len());
                for trigger in triggers {
                    let trigger_type = match trigger.type_ {
                        0 => GrammarTriggerType::Token,
                        1 => GrammarTriggerType::Word,
                        2 => GrammarTriggerType::Pattern,
                        3 => GrammarTriggerType::PatternFull,
                        _ => return Err(ApplyChatTemplateError::InvalidGrammarTriggerType),
                    };
                    let value = if trigger.value.is_null() {
                        return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
                    } else {
                        let bytes = unsafe { CStr::from_ptr(trigger.value) }.to_bytes().to_vec();
                        String::from_utf8(bytes)?
                    };
                    let token = if trigger_type == GrammarTriggerType::Token {
                        Some(LlamaToken(trigger.token))
                    } else {
                        None
                    };
                    parsed.push(GrammarTrigger {
                        trigger_type,
                        value,
                        token,
                    });
                }
                parsed
            };
            let preserved_tokens = if raw_result.preserved_tokens_count == 0 {
                Vec::new()
            } else if raw_result.preserved_tokens.is_null() {
                return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
            } else {
                let tokens = unsafe {
                    slice::from_raw_parts(
                        raw_result.preserved_tokens,
                        raw_result.preserved_tokens_count,
                    )
                };
                let mut parsed = Vec::with_capacity(tokens.len());
                for token in tokens {
                    if token.is_null() {
                        return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
                    }
                    let bytes = unsafe { CStr::from_ptr(*token) }.to_bytes().to_vec();
                    parsed.push(String::from_utf8(bytes)?);
                }
                parsed
            };
            let additional_stops = if raw_result.additional_stops_count == 0 {
                Vec::new()
            } else if raw_result.additional_stops.is_null() {
                return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
            } else {
                let stops = unsafe {
                    slice::from_raw_parts(
                        raw_result.additional_stops,
                        raw_result.additional_stops_count,
                    )
                };
                let mut parsed = Vec::with_capacity(stops.len());
                for stop in stops {
                    if stop.is_null() {
                        return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
                    }
                    let bytes = unsafe { CStr::from_ptr(*stop) }.to_bytes().to_vec();
                    parsed.push(String::from_utf8(bytes)?);
                }
                parsed
            };
            let parse_tool_calls = tools_json.is_some_and(|tools| !tools.is_empty());
            let (thinking_start_tag, thinking_end_tags) =
                unsafe { thinking_metadata_from_raw(&raw_result) }?;
            Ok(ChatTemplateResult {
                prompt,
                grammar,
                grammar_lazy,
                grammar_triggers,
                preserved_tokens,
                additional_stops,
                chat_format: raw_result.chat_format,
                parser,
                generation_prompt,
                parse_tool_calls,
                supports_thinking: raw_result.supports_thinking,
                thinking_start_tag,
                thinking_end_tags,
            })
        })();

        unsafe { llama_cpp_sys_2::llama_rs_chat_template_result_free(&mut raw_result) };
        result
    }

    /// Apply the model chat template using OpenAI-compatible JSON messages.
    #[cfg(feature = "common")]
    #[tracing::instrument(skip_all)]
    pub fn apply_chat_template_oaicompat(
        &self,
        tmpl: &LlamaChatTemplate,
        params: &OpenAIChatTemplateParams<'_>,
    ) -> Result<ChatTemplateResult, ApplyChatTemplateError> {
        let parse_tool_calls = params.parse_tool_calls;
        let messages_cstr = CString::new(params.messages_json)?;
        let tools_cstr = params.tools_json.map(CString::new).transpose()?;
        let tool_choice_cstr = params.tool_choice.map(CString::new).transpose()?;
        let json_schema_cstr = params.json_schema.map(CString::new).transpose()?;
        let grammar_cstr = params.grammar.map(CString::new).transpose()?;
        let reasoning_cstr = params.reasoning_format.map(CString::new).transpose()?;
        let kwargs_cstr = params.chat_template_kwargs.map(CString::new).transpose()?;

        let mut raw_result = llama_cpp_sys_2::llama_rs_chat_template_result {
            prompt: ptr::null_mut(),
            grammar: ptr::null_mut(),
            parser: ptr::null_mut(),
            generation_prompt: ptr::null_mut(),
            chat_format: 0,
            grammar_lazy: false,
            grammar_triggers: ptr::null_mut(),
            grammar_triggers_count: 0,
            preserved_tokens: ptr::null_mut(),
            preserved_tokens_count: 0,
            additional_stops: ptr::null_mut(),
            additional_stops_count: 0,
            supports_thinking: false,
            thinking_start_tag: ptr::null_mut(),
            thinking_end_tags: ptr::null_mut(),
            thinking_end_tags_count: 0,
        };

        let ffi_params = llama_cpp_sys_2::llama_rs_chat_template_oaicompat_params {
            messages: messages_cstr.as_ptr(),
            tools: tools_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            tool_choice: tool_choice_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            json_schema: json_schema_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            grammar: grammar_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            reasoning_format: reasoning_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            chat_template_kwargs: kwargs_cstr
                .as_ref()
                .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            add_generation_prompt: params.add_generation_prompt,
            use_jinja: params.use_jinja,
            parallel_tool_calls: params.parallel_tool_calls,
            enable_thinking: params.enable_thinking,
            add_bos: params.add_bos,
            add_eos: params.add_eos,
        };

        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_apply_chat_template_oaicompat(
                self.model.as_ptr(),
                tmpl.0.as_ptr(),
                &ffi_params,
                &mut raw_result,
            )
        };

        let result = (|| {
            if !status_is_ok(rc) {
                return Err(match crate::status_error_message(rc) {
                    Some(message) => ApplyChatTemplateError::FfiException(rc, message),
                    None => ApplyChatTemplateError::FfiError(rc),
                });
            }
            if raw_result.prompt.is_null() {
                return Err(ApplyChatTemplateError::NullResult);
            }
            let prompt_bytes = unsafe { CStr::from_ptr(raw_result.prompt) }
                .to_bytes()
                .to_vec();
            let prompt = String::from_utf8(prompt_bytes)?;
            let grammar_lazy = raw_result.grammar_lazy;
            let grammar = if raw_result.grammar.is_null() {
                None
            } else {
                let grammar_bytes = unsafe { CStr::from_ptr(raw_result.grammar) }
                    .to_bytes()
                    .to_vec();
                Some(String::from_utf8(grammar_bytes)?)
            };
            let parser = NonNull::new(std::mem::replace(&mut raw_result.parser, ptr::null_mut()))
                .map(|handle| ChatParser(std::sync::Arc::new(ChatParserHandle(handle))));
            let generation_prompt = if raw_result.generation_prompt.is_null() {
                String::new()
            } else {
                let generation_prompt_bytes =
                    unsafe { CStr::from_ptr(raw_result.generation_prompt) }
                        .to_bytes()
                        .to_vec();
                String::from_utf8(generation_prompt_bytes)?
            };
            let grammar_triggers = if raw_result.grammar_triggers_count == 0 {
                Vec::new()
            } else if raw_result.grammar_triggers.is_null() {
                return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
            } else {
                let triggers = unsafe {
                    slice::from_raw_parts(
                        raw_result.grammar_triggers,
                        raw_result.grammar_triggers_count,
                    )
                };
                let mut parsed = Vec::with_capacity(triggers.len());
                for trigger in triggers {
                    let trigger_type = match trigger.type_ {
                        0 => GrammarTriggerType::Token,
                        1 => GrammarTriggerType::Word,
                        2 => GrammarTriggerType::Pattern,
                        3 => GrammarTriggerType::PatternFull,
                        _ => return Err(ApplyChatTemplateError::InvalidGrammarTriggerType),
                    };
                    let value = if trigger.value.is_null() {
                        String::new()
                    } else {
                        let bytes = unsafe { CStr::from_ptr(trigger.value) }.to_bytes().to_vec();
                        String::from_utf8(bytes)?
                    };
                    let token = if trigger_type == GrammarTriggerType::Token {
                        Some(LlamaToken(trigger.token))
                    } else {
                        None
                    };
                    parsed.push(GrammarTrigger {
                        trigger_type,
                        value,
                        token,
                    });
                }
                parsed
            };
            let preserved_tokens = if raw_result.preserved_tokens_count == 0 {
                Vec::new()
            } else if raw_result.preserved_tokens.is_null() {
                return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
            } else {
                let tokens = unsafe {
                    slice::from_raw_parts(
                        raw_result.preserved_tokens,
                        raw_result.preserved_tokens_count,
                    )
                };
                let mut parsed = Vec::with_capacity(tokens.len());
                for token in tokens {
                    if token.is_null() {
                        return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
                    }
                    let bytes = unsafe { CStr::from_ptr(*token) }.to_bytes().to_vec();
                    parsed.push(String::from_utf8(bytes)?);
                }
                parsed
            };
            let additional_stops = if raw_result.additional_stops_count == 0 {
                Vec::new()
            } else if raw_result.additional_stops.is_null() {
                return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
            } else {
                let stops = unsafe {
                    slice::from_raw_parts(
                        raw_result.additional_stops,
                        raw_result.additional_stops_count,
                    )
                };
                let mut parsed = Vec::with_capacity(stops.len());
                for stop in stops {
                    if stop.is_null() {
                        return Err(ApplyChatTemplateError::InvalidGrammarTriggerType);
                    }
                    let bytes = unsafe { CStr::from_ptr(*stop) }.to_bytes().to_vec();
                    parsed.push(String::from_utf8(bytes)?);
                }
                parsed
            };

            let (thinking_start_tag, thinking_end_tags) =
                unsafe { thinking_metadata_from_raw(&raw_result) }?;
            Ok(ChatTemplateResult {
                prompt,
                grammar,
                grammar_lazy,
                grammar_triggers,
                preserved_tokens,
                additional_stops,
                chat_format: raw_result.chat_format,
                parser,
                generation_prompt,
                parse_tool_calls,
                supports_thinking: raw_result.supports_thinking,
                thinking_start_tag,
                thinking_end_tags,
            })
        })();

        unsafe { llama_cpp_sys_2::llama_rs_chat_template_result_free(&mut raw_result) };
        result
    }
}

#[cfg(feature = "common")]
impl ChatTemplateResult {
    /// Parse a generated response into an OpenAI-compatible message JSON string.
    pub fn parse_response_oaicompat(
        &self,
        text: &str,
        is_partial: bool,
    ) -> Result<String, ChatParseError> {
        let text_cstr = CString::new(text)?;
        let generation_prompt_cstr = if self.generation_prompt.is_empty() {
            None
        } else {
            Some(CString::new(self.generation_prompt.as_str())?)
        };
        let mut out_json: *mut c_char = ptr::null_mut();
        let rc = unsafe {
            llama_cpp_sys_2::llama_rs_chat_parse_to_oaicompat(
                text_cstr.as_ptr(),
                is_partial,
                self.chat_format,
                self.parser.as_ref().map_or(ptr::null(), ChatParser::as_ptr),
                generation_prompt_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
                &mut out_json,
            )
        };

        let result = (|| {
            if !status_is_ok(rc) {
                return Err(match crate::status_error_message(rc) {
                    Some(message) => ChatParseError::FfiException(rc, message),
                    None => ChatParseError::FfiError(rc),
                });
            }
            if out_json.is_null() {
                return Err(ChatParseError::NullResult);
            }
            let bytes = unsafe { CStr::from_ptr(out_json) }.to_bytes().to_vec();
            Ok(String::from_utf8(bytes)?)
        })();

        unsafe { llama_cpp_sys_2::llama_rs_string_free(out_json) };
        result
    }

    /// Initialize a streaming parser for OpenAI-compatible chat deltas.
    pub fn streaming_state_oaicompat(&self) -> Result<ChatParseStateOaicompat, ChatParseError> {
        let generation_prompt_cstr = if self.generation_prompt.is_empty() {
            None
        } else {
            Some(CString::new(self.generation_prompt.as_str())?)
        };
        let state = unsafe {
            llama_cpp_sys_2::llama_rs_chat_parse_state_init_oaicompat(
                self.chat_format,
                self.parser.as_ref().map_or(ptr::null(), ChatParser::as_ptr),
                generation_prompt_cstr
                    .as_ref()
                    .map_or(ptr::null(), |cstr| cstr.as_ptr()),
            )
        };
        let state = NonNull::new(state).ok_or(ChatParseError::NullResult)?;
        Ok(ChatParseStateOaicompat { state })
    }
}

/// Generic helper function for extracting string values from the C API
/// This are specifically useful for the the metadata functions, where we pass in a buffer
/// to be populated by a string, not yet knowing if the buffer is large enough.
/// If the buffer was not large enough, we get the correct length back, which can be used to
/// construct a buffer of appropriate size.
fn extract_meta_string<F>(c_function: F, capacity: usize) -> Result<String, MetaValError>
where
    F: Fn(*mut c_char, usize) -> i32,
{
    let mut buffer = vec![0u8; capacity];

    // call the foreign function
    let result = c_function(buffer.as_mut_ptr().cast::<c_char>(), buffer.len());
    if result < 0 {
        return Err(MetaValError::NegativeReturn(result));
    }

    // check if the response fit in our buffer
    let returned_len = result as usize;
    if returned_len >= capacity {
        // buffer wasn't large enough, try again with the correct capacity.
        return extract_meta_string(c_function, returned_len + 1);
    }

    // verify null termination
    debug_assert_eq!(
        buffer.get(returned_len),
        Some(&0),
        "should end with null byte"
    );

    // resize, convert, and return
    buffer.truncate(returned_len);
    Ok(String::from_utf8(buffer)?)
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_model_free(self.model.as_mut_ptr()) }
    }
}
