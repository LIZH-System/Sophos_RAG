# Configuration Management

This directory contains configuration files for the Sophos RAG system. To ensure security, sensitive information like API keys and proxy settings are managed through environment variables.

## Setup Instructions

1. Copy the template configuration file:
   ```bash
   cp config_template.yaml config.yaml
   ```

2. Set up your environment variables:
   ```bash
   # Required
   export DEEPSEEK_API_KEY="your-api-key-here"
   
   # Optional (if using proxy)
   export HTTP_PROXY="http://your-proxy-server:port"
   export HTTPS_PROXY="http://your-proxy-server:port"
   ```

3. Edit `config.yaml` to customize your settings:
   - Enable/disable proxy by setting `proxy.enabled`
   - Adjust model parameters in the `deepseek` section
   - Configure logging settings in the `logging` section

## Security Notes

- Never commit `config.yaml` to version control
- Keep your API keys and proxy settings in environment variables
- Use different API keys for development and production
- Regularly rotate your API keys

## Configuration Options

### DeepSeek API Settings
- `api_key`: Your DeepSeek API key (from environment variable)
- `model_name`: The model to use (default: deepseek-chat)
- `temperature`: Controls randomness (0.0 to 1.0)
- `max_tokens`: Maximum tokens in response
- `top_p`: Nucleus sampling parameter
- `base_url`: API endpoint
- `verify_ssl`: SSL certificate verification
- `max_retries`: Number of retry attempts
- `timeout`: Request timeout in seconds

### Proxy Settings
- `enabled`: Enable/disable proxy
- `http`: HTTP proxy URL
- `https`: HTTPS proxy URL

### Logging Settings
- `level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `format`: Log message format 