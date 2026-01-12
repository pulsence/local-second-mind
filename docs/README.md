# Local Second Mind Documentation

Welcome to the Local Second Mind (LSM) documentation! This guide will help you understand, use, and extend LSM - a local-first retrieval-augmented generation (RAG) system for personal knowledge management.

## Documentation Structure

### 📘 User Guide
Start here if you're new to LSM or want to learn how to use its features.

- [Getting Started](user-guide/GETTING_STARTED.md) - Installation and first steps
- [Configuration Guide](user-guide/CONFIGURATION.md) - Complete configuration reference
- [CLI Usage](user-guide/CLI_USAGE.md) - Command-line interface guide
- [Query Modes](user-guide/QUERY_MODES.md) - Understanding query modes (grounded, insight, hybrid)
- [Notes System](user-guide/NOTES.md) - Saving and organizing query results
- [Remote Sources](user-guide/REMOTE_SOURCES.md) - Web search integration

### 🏗️ Architecture
Learn about LSM's internal design and architecture.

- [System Overview](architecture/OVERVIEW.md) - High-level architecture
- [Ingest Pipeline](architecture/INGEST.md) - Document processing and embedding
- [Query Pipeline](architecture/QUERY.md) - Retrieval and synthesis
- [Provider System](architecture/PROVIDERS.md) - LLM provider abstraction
- [Mode System](architecture/MODES.md) - Source policies and modes

### 📚 API Reference
Detailed API documentation for developers.

- [Configuration API](api-reference/CONFIG.md) - Configuration dataclasses
- [Provider API](api-reference/PROVIDERS.md) - LLM provider interface
- [Remote API](api-reference/REMOTE.md) - Remote source providers
- [REPL Commands](api-reference/REPL.md) - Interactive command reference

### 🛠️ Development
Contributing to LSM development.

- [Development Setup](development/SETUP.md) - Setting up a development environment
- [Testing Guide](development/TESTING.md) - Running and writing tests
- [Adding Providers](ADDING_PROVIDERS.md) - How to add new LLM providers
- [Contributing](development/CONTRIBUTING.md) - Contribution guidelines
- [Changelog](development/CHANGELOG.md) - Version history

## Quick Links

### For Users
- **First time user?** Start with [Getting Started](user-guide/GETTING_STARTED.md)
- **Need help with commands?** See [CLI Usage](user-guide/CLI_USAGE.md)
- **Want to configure LSM?** Check [Configuration Guide](user-guide/CONFIGURATION.md)

### For Developers
- **Want to add a provider?** See [Adding Providers](ADDING_PROVIDERS.md)
- **Setting up dev environment?** Check [Development Setup](development/SETUP.md)
- **Writing tests?** Read [Testing Guide](development/TESTING.md)

### For Architects
- **Understanding the system?** Start with [System Overview](architecture/OVERVIEW.md)
- **Want to understand modes?** See [Mode System](architecture/MODES.md)
- **Looking at extensibility?** Check [Provider System](architecture/PROVIDERS.md)

## Version Information

- **Current Version:** 0.1.0
- **Python Version:** 3.10+
- **Documentation Status:** ✅ Complete

## Getting Help

- **Issues:** Open an issue on GitHub
- **Discussions:** Join our GitHub Discussions
- **Documentation:** You're already here!

## License

MIT License - see LICENSE file for details
