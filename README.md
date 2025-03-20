# Guaxinim - Your Personal Coffee Assistant 🦝☕

Guaxinim is an AI-powered coffee assistant that helps you brew the perfect cup of coffee. It provides brewing guides, improvement suggestions, and answers to your coffee-related questions.

## Features

- **Coffee Brewing Guides**: Get detailed instructions for various brewing methods (V60, French Press, Espresso, etc.)
- **Coffee Improvement Suggestions**: Input your current coffee parameters and get personalized suggestions
- **Coffee Knowledge Base**: Ask questions about coffee and get expert answers
- **Custom Knowledge Sources**: Add your own coffee knowledge through JSON files
- **Smart Caching**: Redis-based caching system to optimize API usage and response times

## Adding Custom Knowledge Sources

Guaxinim can learn from both YouTube videos and PDF documents. Here's how to extend its knowledge base:

1. You can use the pre-configured YouTube channels list in `config/youtube_channels.json`, which includes popular coffee experts like James Hoffmann and European Coffee Trip. Or create your own channels list:
```json
{
    "channels": [
        "jameshoffmann",
        "europeancoffeetrip",
        "LanceHedrick"
    ]
}
```

2. Place your PDF documents in a directory (e.g., `data/raw/pdf/`)

3. Run the knowledge base initialization script:
```bash
# Using the pre-configured channels list
python scripts/initialize_knowledge_base.py \
    --channels-file config/youtube_channels.json \
    --pdf-dir data/raw/pdf

# Or using your custom channels list
python scripts/initialize_knowledge_base.py \
    --channels-file your_channels.json \
    --pdf-dir data/raw/pdf
```

This script will:
- Crawl and download videos from the specified YouTube channels
- Process video transcripts into searchable content
- Convert PDF documents into searchable text
- Generate embeddings for all processed content

Your knowledge will be available in the next Streamlit session!

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lfomendes/guaxinim-barista.git
cd guaxinim-barista
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
virtualenv venv 

# Activate virtual environment
source venv/bin/activate
```

3. Set up environment variables:
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your actual values
# Make sure to add your OpenAI API key
vim .env
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the application:
```bash
streamlit run main.py
```

## Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY`: Your OpenAI API key for generating coffee recommendations

## Caching System

Guaxinim uses Redis for persistent caching of API responses to improve performance and reduce API costs:

- **Cache Duration**: Responses are cached for 30 days by default
- **Cache Control**: Use the 🧹 Clear Cache button in the sidebar to manually clear the cache
- **Cache Keys**: Responses are cached based on:
  - Function name
  - Input parameters
  - Similarity search settings
  - Bot configuration

### Redis Setup

1. Install Redis:
```bash
sudo apt-get install redis-server
```

2. Verify Redis is running:
```bash
systemctl status redis-server
```

3. Redis configuration is automatically handled by the application

To set up your environment:
1. Copy `env.example` to `.env`
2. Replace the placeholder values with your actual API keys
3. Never commit your `.env` file to version control

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Quality
Run the quality checks:
```bash
./run_quality_checks.sh
```

## Utility Scripts

Guaxinim provides a main script to initialize and update its knowledge base:

### Knowledge Base Initialization
```bash
# Initialize or update the knowledge base
python scripts/initialize_knowledge_base.py \
    --channels-file channels.json \
    --pdf-dir data/raw/pdf
```

This script automates the entire process of:
1. Downloading and processing YouTube videos
2. Converting PDFs into searchable text
3. Creating embeddings for all content

The script requires a YouTube API key to be set in your `.env` file:
```
YOUTUBE_API_KEY=your_api_key_here
```

To get a YouTube API key:
1. Go to the [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Create credentials (API key)
5. Copy the API key and paste it in your `.env` file

### Directory Structure
```
data/
├── raw/
│   ├── pdf/            # Place your PDF files here
│   ├── transcripts/     # YouTube transcripts (auto-generated)
│   ├── pdf_processed/   # Processed PDF content
│   └── youtube_processed/ # Processed YouTube content
└── embeddings/         # Generated embeddings
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
