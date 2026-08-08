use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{ChatTemplateResult, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::openai::OpenAIChatTemplateParams;

const HISTORY: &str = r"
{%- for message in messages %}
    {%- if message.role == 'assistant' %}
        {{- '<|im_start|>assistant\n<think>\n' + (message.reasoning_content or '') + '\n</think>\n\n' + message.content + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
";

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

const SELF_OPENING_TAIL: &str = r"
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
";

const MESSAGES: &str = r#"[{"role":"user","content":"Hi"}]"#;

fn vocab_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("LLAMA_TEST_VOCAB_GGUF") {
        return Some(PathBuf::from(path));
    }
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../llama-cpp-sys-2/llama.cpp/models/ggml-vocab-qwen35.gguf");
    bundled.exists().then_some(bundled)
}

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
