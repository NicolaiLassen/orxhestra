"""Tests for content parser — parse_content_blocks."""

from orxhestra.models.content_parser import parse_content_blocks


class TestParseContentBlocks:
    def test_plain_string(self):
        assert parse_content_blocks("hello world") == ("hello world", "")

    def test_empty_string(self):
        assert parse_content_blocks("") == ("", "")

    def test_empty_list(self):
        assert parse_content_blocks([]) == ("", "")

    def test_text_block(self):
        content = [{"type": "text", "text": "Hello"}]
        assert parse_content_blocks(content) == ("Hello", "")

    def test_multiple_text_blocks(self):
        content = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]
        assert parse_content_blocks(content) == ("Hello world", "")

    def test_anthropic_thinking(self):
        content = [
            {"type": "thinking", "thinking": "Let me think..."},
            {"type": "text", "text": "The answer is 42."},
        ]
        text, thinking = parse_content_blocks(content)
        assert text == "The answer is 42."
        assert thinking == "Let me think..."

    def test_response_api_reasoning_with_nested_summary(self):
        content = [
            {
                "type": "reasoning",
                "id": "rs_123",
                "summary": [
                    {"type": "summary_text", "text": "First I considered..."},
                    {"type": "summary_text", "text": " Then I decided..."},
                ],
            },
            {"type": "text", "text": "The answer is 42."},
        ]
        text, thinking = parse_content_blocks(content)
        assert text == "The answer is 42."
        assert thinking == "First I considered... Then I decided..."

    def test_response_api_reasoning_empty_summary(self):
        content = [{"type": "reasoning", "id": "rs_123", "summary": []}]
        assert parse_content_blocks(content) == ("", "")

    def test_v1_standard_reasoning_block(self):
        """LangChain v1 standardized reasoning block (all providers post-translation)."""
        content = [
            {"type": "reasoning", "reasoning": "Let me think..."},
            {"type": "text", "text": "The answer is 42."},
        ]
        text, thinking = parse_content_blocks(content)
        assert text == "The answer is 42."
        assert thinking == "Let me think..."

    def test_v1_standard_reasoning_multiple_blocks(self):
        """OpenAI Responses `_explode_reasoning` emits one block per summary part."""
        content = [
            {"type": "reasoning", "reasoning": "First I considered..."},
            {"type": "reasoning", "reasoning": " Then I decided..."},
            {"type": "text", "text": "Done."},
        ]
        text, thinking = parse_content_blocks(content)
        assert text == "Done."
        assert thinking == "First I considered... Then I decided..."

    def test_bedrock_converse_reasoning_content(self):
        """Bedrock Converse raw reasoning block before v1 translation."""
        content = [
            {
                "type": "reasoning_content",
                "reasoning_content": {
                    "text": "Analyzing the request...",
                    "signature": "sig_abc",
                },
            },
            {"type": "text", "text": "Here is the answer."},
        ]
        text, thinking = parse_content_blocks(content)
        assert text == "Here is the answer."
        assert thinking == "Analyzing the request..."

    def test_bedrock_converse_reasoning_content_missing(self):
        content = [{"type": "reasoning_content"}]
        assert parse_content_blocks(content) == ("", "")

    def test_function_call_skipped(self):
        content = [
            {"type": "text", "text": "Let me search."},
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "search",
                "arguments": '{"q": "test"}',
            },
        ]
        text, thinking = parse_content_blocks(content)
        assert text == "Let me search."
        assert thinking == ""

    def test_web_search_call_skipped(self):
        content = [
            {"type": "web_search_call", "id": "ws_123", "status": "completed"},
            {"type": "text", "text": "Found it."},
        ]
        assert parse_content_blocks(content) == ("Found it.", "")

    def test_refusal_skipped(self):
        content = [{"type": "refusal", "refusal": "I cannot do that."}]
        assert parse_content_blocks(content) == ("", "")

    def test_mixed_response_api(self):
        """Full Response API response with text, reasoning, function_call."""
        content = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Thinking..."}],
            },
            {"type": "text", "text": "Here's the result.", "id": "msg_1"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Copenhagen"}',
            },
        ]
        text, thinking = parse_content_blocks(content)
        assert text == "Here's the result."
        assert thinking == "Thinking..."

    def test_non_dict_list_items(self):
        content = ["plain string item", 42]
        text, thinking = parse_content_blocks(content)
        assert text == "plain string item42"
        assert thinking == ""

    def test_block_without_type(self):
        content = [{"text": "no type key"}]
        # type defaults to "", doesn't match any handler, skipped
        assert parse_content_blocks(content) == ("", "")
