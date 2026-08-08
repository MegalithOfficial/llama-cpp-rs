//! Guards the reasoning-channel contract of `apply_chat_template_oaicompat`.
//!
//! Templates in the Qwen3-Thinking and DeepSeek-R1 families put the opening
//! `<think>` in the generation prompt, so the model resumes inside the
//! reasoning channel and only emits the closing tag. That prefill has to
//! survive into the rendered prompt: without it the model is never nudged into
//! the reasoning channel and reasoning leaks into the answer content. Upstream
//! merges rewrite the prompt-assembly path often enough that this deserves a
//! test rather than trust.
//!
//! These run against the vocabulary bundled with the `llama.cpp` submodule.
//! Override it with `LLAMA_TEST_VOCAB_GGUF`; the tests skip when neither is
//! available.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{ChatTemplateResult, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;

/// Renders assistant turns with an explicit `<think>` block, which is what lets
/// the differential autoparser recognise a reasoning channel at all.
const HISTORY: &str = r"
{%- for message in messages %}
    {%- if message.role == 'assistant' %}
        {{- '<|im_start|>assistant\n<think>\n' + (message.reasoning_content or '') + '\n</think>\n\n' + message.content + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
";

/// Qwen3-Thinking / Qwen3.5-Thinking generation-prompt shape: `<think>` is
/// forced open unless thinking is explicitly turned off.
const FORCED_OPEN_TAIL: &str = r"
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
";

/// Original Qwen3 shape: the model opens its own `<think>`, so the generation
/// prompt carries a tag only when thinking is turned off.
const SELF_OPENING_TAIL: &str = r"
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
";

const MESSAGES: &str = r#"[{"role":"user","content":"Hi"}]"#;

/// `LLAMA_TEST_VOCAB_GGUF` when set, else the vocabulary bundled with the
/// submodule, else `None` so the tests skip instead of failing.
fn vocab_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("LLAMA_TEST_VOCAB_GGUF") {
        return Some(PathBuf::from(path));
    }
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../llama-cpp-sys-2/llama.cpp/models/ggml-vocab-qwen35.gguf");
    bundled.exists().then_some(bundled)
}

/// The vocabulary from [`vocab_path`], or `None` to skip.
///
/// The backend can be started only one time in a process, and the tests share
/// this process, so both the backend and the model are made one time here.
/// `vocab_only` keeps the load cheap: rendering an explicit template needs the
/// bos/eos tokens and no tensor data.
fn model() -> Option<&'static LlamaModel> {
    static MODEL: OnceLock<Option<(LlamaBackend, LlamaModel)>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let path = vocab_path()?;
            let backend = LlamaBackend::init().unwrap();
            let params = LlamaModelParams::default().with_vocab_only(true);
            let model = LlamaModel::load_from_file(&backend, path, &params).unwrap();
            Some((backend, model))
        })
        .as_ref()
        .map(|(_backend, model)| model)
}

fn render(model: &LlamaModel, tail: &str, enable_thinking: bool) -> ChatTemplateResult {
    let template = LlamaChatTemplate::new(&format!("{HISTORY}{tail}")).unwrap();
    let params = OpenAIChatTemplateParams {
        messages_json: MESSAGES,
        tools_json: None,
        tool_choice: None,
        json_schema: None,
        grammar: None,
        reasoning_format: Some("auto"),
        chat_template_kwargs: None,
        add_generation_prompt: true,
        use_jinja: true,
        parallel_tool_calls: false,
        enable_thinking,
        add_bos: false,
        add_eos: false,
        parse_tool_calls: false,
    };
    model
        .apply_chat_template_oaicompat(&template, &params)
        .unwrap()
}

#[test]
fn forced_open_think_survives_into_the_prompt() {
    let Some(model) = model() else {
        return;
    };
    let result = render(model, FORCED_OPEN_TAIL, true);

    assert!(
        result.prompt.ends_with("<|im_start|>assistant\n<think>\n"),
        "forced-open <think> prefill missing from prompt: {:?}",
        result.prompt
    );
    assert_eq!(result.generation_prompt, "<|im_start|>assistant\n<think>\n");
    assert!(result.thinking_forced_open());
}

#[test]
fn disabled_thinking_pre_closes_the_channel() {
    let Some(model) = model() else {
        return;
    };
    let result = render(model, FORCED_OPEN_TAIL, false);

    assert!(
        result.prompt.ends_with("<think>\n\n</think>\n\n"),
        "pre-closed think block missing from prompt: {:?}",
        result.prompt
    );
    assert!(!result.thinking_forced_open());
}

#[test]
fn self_opening_template_gets_no_prefill() {
    let Some(model) = model() else {
        return;
    };
    let result = render(model, SELF_OPENING_TAIL, true);

    assert!(
        result.prompt.ends_with("<|im_start|>assistant\n"),
        "unexpected trailing prefill: {:?}",
        result.prompt
    );
    assert!(
        !result.prompt.contains("<think>"),
        "template opens <think> itself, the prompt must not pre-open it: {:?}",
        result.prompt
    );
    assert!(!result.thinking_forced_open());
}

#[test]
fn reasoning_tags_are_reported() {
    let Some(model) = model() else {
        return;
    };
    let result = render(model, FORCED_OPEN_TAIL, true);

    assert!(result.supports_thinking);
    assert_eq!(result.thinking_start_tag.as_deref(), Some("<think>"));
    assert!(result.thinking_end_tags.iter().any(|tag| tag == "</think>"));
}
