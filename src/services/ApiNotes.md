# Services API Notes

## Overview

The services directory contains modules that provide external integrations and shared
functionality across the stylometrics application. These services abstract away the
complexities of external APIs and provide unified interfaces for the rest of the application.

## LLMApi Service

The LLMApi service provides a vendor-agnostic interface for interacting with various
Language Learning Model providers (OpenAI, Anthropic, etc.).

### Design Goals

1. Abstract provider-specific implementation details
2. Provide a consistent interface for all LLM interactions
3. Support both chat-based and completion-based interfaces
4. Handle rate limiting, retries, and error normalization

### API Stability

The LLMApi is considered an internal API but should maintain backwards compatibility
for major versions. Breaking changes should be clearly documented.
