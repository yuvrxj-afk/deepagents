<!-- markdownlint-disable MD024 -->

# Changelog

## [0.1.5](https://github.com/yuvrxj-afk/deepagents/compare/langchain-quickjs==0.1.4...langchain-quickjs==0.1.5) (2026-06-05)


### Features

* **quickjs:** add `max_ptc_calls` budget for ptc calls ([#2994](https://github.com/yuvrxj-afk/deepagents/issues/2994)) ([13f6c2d](https://github.com/yuvrxj-afk/deepagents/commit/13f6c2dec448e270d7e685447292d394d31fe528))
* **quickjs:** add REPL persistence modes ([#3557](https://github.com/yuvrxj-afk/deepagents/issues/3557)) ([0cda6f3](https://github.com/yuvrxj-afk/deepagents/commit/0cda6f3ab28bc83cd16ec9fcc48229bdf6f2dc1a))
* **quickjs:** add snapshot-based repl persistence between turns ([#3064](https://github.com/yuvrxj-afk/deepagents/issues/3064)) ([c46feed](https://github.com/yuvrxj-afk/deepagents/commit/c46feed66e83876054489a8454904de4c87ddf6b))
* **quickjs:** add swarm task tool ([#3472](https://github.com/yuvrxj-afk/deepagents/issues/3472)) ([2c28b7b](https://github.com/yuvrxj-afk/deepagents/commit/2c28b7b8c2ac7571fc3a1f0d8d00f5697fe3e90e))
* **quickjs:** propagate return types ([#3210](https://github.com/yuvrxj-afk/deepagents/issues/3210)) ([e26bccb](https://github.com/yuvrxj-afk/deepagents/commit/e26bccbe81b4e3ff2f0332f56f683106e0bafd88))
* **quickjs:** rename middleware ([#3334](https://github.com/yuvrxj-afk/deepagents/issues/3334)) ([fc80075](https://github.com/yuvrxj-afk/deepagents/commit/fc80075c65c3b4beb8f672b6bb27464fee6d79c2))
* **quickjs:** surface tool exceptions as the original error ([#3049](https://github.com/yuvrxj-afk/deepagents/issues/3049)) ([d96dc8c](https://github.com/yuvrxj-afk/deepagents/commit/d96dc8ce4a1d2d25abc606266cfe14629789c457))
* **sdk:** surface subagents via inherited `lc_agent_name` projection ([e0a1ed2](https://github.com/yuvrxj-afk/deepagents/commit/e0a1ed24e6b44c31d0aac3358aeee0d6cb66b2c4))
* **sdk:** v0.6 ([4db09ac](https://github.com/yuvrxj-afk/deepagents/commit/4db09acba34b38521192b8f278723524be560779))


### Bug Fixes

* **quickjs:** add `ls_code_input_language` metadata to `eval` tool ([#3062](https://github.com/yuvrxj-afk/deepagents/issues/3062)) ([b9bc674](https://github.com/yuvrxj-afk/deepagents/commit/b9bc674fe67ac40e000b30bbb4a753a9b6e167ed))
* **quickjs:** auto-await final-expression Promise in eval REPL ([#3499](https://github.com/yuvrxj-afk/deepagents/issues/3499)) ([f7f894a](https://github.com/yuvrxj-afk/deepagents/commit/f7f894aa9f313cf8157bc6d7711013f5509d0b46))
* **quickjs:** bound console buffering at capture time ([#2999](https://github.com/yuvrxj-afk/deepagents/issues/2999)) ([251e405](https://github.com/yuvrxj-afk/deepagents/commit/251e405d8b1897ae09ee47b0a8be48edfaaad8a1))
* **quickjs:** handle top-level await in snapshot step ([#3161](https://github.com/yuvrxj-afk/deepagents/issues/3161)) ([b330c22](https://github.com/yuvrxj-afk/deepagents/commit/b330c222ee661f7944a59d7b0ef2e0c7a42b1e29))
* **quickjs:** keep PTC loop and runtime context in `_PTCState` ([#3134](https://github.com/yuvrxj-afk/deepagents/issues/3134)) ([c70cc5a](https://github.com/yuvrxj-afk/deepagents/commit/c70cc5a602bd26191f347dd31cc4e5d17c75f65f))
* **quickjs:** remove ptc command buffering ([#3023](https://github.com/yuvrxj-afk/deepagents/issues/3023)) ([ac1218a](https://github.com/yuvrxj-afk/deepagents/commit/ac1218a1b14cde2f066c59804ab86567821f8c9d))
* **quickjs:** scope REPL prompt sandbox bullet to the runtime ([#3528](https://github.com/yuvrxj-afk/deepagents/issues/3528)) ([1b395ab](https://github.com/yuvrxj-afk/deepagents/commit/1b395ab9699b1f384a85efeeef732ea7e4fc523a))
* **quickjs:** swarm subagent doesn't allow configuring middleware ([#3757](https://github.com/yuvrxj-afk/deepagents/issues/3757)) ([3394a9d](https://github.com/yuvrxj-afk/deepagents/commit/3394a9d9c7c89c0a28fa1328c9f6bae68a83ff14))
* **quickjs:** update system prompt snapshots ([#3450](https://github.com/yuvrxj-afk/deepagents/issues/3450)) ([9f9220d](https://github.com/yuvrxj-afk/deepagents/commit/9f9220d80737208faa9262c0bdfb3eeafc0e13c8))
* **sdk:** stable `HumanMessage` IDs across resumed threads ([#3591](https://github.com/yuvrxj-afk/deepagents/issues/3591)) ([82c3194](https://github.com/yuvrxj-afk/deepagents/commit/82c31947f9dc938ffc71e1cea96d162a39aec3a1))


### Reverted Changes

* **quickjs:** release: 0.1.1 ([#3255](https://github.com/yuvrxj-afk/deepagents/issues/3255)) ([8125f71](https://github.com/yuvrxj-afk/deepagents/commit/8125f71a6ffd40b75a25c017e2b255eeb3be48a6))

## [0.1.4](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.1.3...langchain-quickjs==0.1.4) (2026-06-03)

### Bug Fixes

* Swarm subagent doesn't allow configuring middleware ([#3757](https://github.com/langchain-ai/deepagents/issues/3757)) ([3394a9d](https://github.com/langchain-ai/deepagents/commit/3394a9d9c7c89c0a28fa1328c9f6bae68a83ff14))

## [0.1.3](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.1.2...langchain-quickjs==0.1.3) (2026-06-01)

### Features

* Add REPL persistence modes ([#3557](https://github.com/langchain-ai/deepagents/issues/3557)) ([0cda6f3](https://github.com/langchain-ai/deepagents/commit/0cda6f3ab28bc83cd16ec9fcc48229bdf6f2dc1a))
* Add swarm task tool ([#3472](https://github.com/langchain-ai/deepagents/issues/3472)) ([2c28b7b](https://github.com/langchain-ai/deepagents/commit/2c28b7b8c2ac7571fc3a1f0d8d00f5697fe3e90e))

### Bug Fixes

* Auto-await final-expression Promise in eval REPL ([#3499](https://github.com/langchain-ai/deepagents/issues/3499)) ([f7f894a](https://github.com/langchain-ai/deepagents/commit/f7f894aa9f313cf8157bc6d7711013f5509d0b46))
* Scope REPL prompt sandbox bullet to the runtime ([#3528](https://github.com/langchain-ai/deepagents/issues/3528)) ([1b395ab](https://github.com/langchain-ai/deepagents/commit/1b395ab9699b1f384a85efeeef732ea7e4fc523a))
* Update system prompt snapshots ([#3450](https://github.com/langchain-ai/deepagents/issues/3450)) ([9f9220d](https://github.com/langchain-ai/deepagents/commit/9f9220d80737208faa9262c0bdfb3eeafc0e13c8))
* Stable `HumanMessage` IDs across resumed threads ([#3591](https://github.com/langchain-ai/deepagents/issues/3591)) ([82c3194](https://github.com/langchain-ai/deepagents/commit/82c31947f9dc938ffc71e1cea96d162a39aec3a1))

## [0.1.2](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.1.1...langchain-quickjs==0.1.2) (2026-05-11)

### Features

* Rename middleware ([#3334](https://github.com/langchain-ai/deepagents/issues/3334)) ([fc80075](https://github.com/langchain-ai/deepagents/commit/fc80075c65c3b4beb8f672b6bb27464fee6d79c2))

## [0.1.1](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.1.0...langchain-quickjs==0.1.1) (2026-05-08)

### Features

* Propagate return types ([#3210](https://github.com/langchain-ai/deepagents/issues/3210)) ([e26bccb](https://github.com/langchain-ai/deepagents/commit/e26bccbe81b4e3ff2f0332f56f683106e0bafd88))

## [0.1.0](https://github.com/langchain-ai/deepagents/compare/langchain-quickjs==0.0.1...langchain-quickjs==0.1.0) (2026-05-05)

This release introduces a new QuickJS runtime implementation backed by `quickjs-rs`.

---

## Prior Releases

Versions prior to 0.0.2 were released without release-please and do not have changelog entries. Refer to the [releases page](https://github.com/langchain-ai/deepagents/releases?q=langchain-quickjs) for details on previous versions.
