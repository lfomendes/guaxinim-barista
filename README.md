# Guaxinim - Your Personal Coffee Assistant 🦝☕

Guaxinim is an AI-powered coffee assistant that helps you brew the perfect cup of coffee. It provides brewing guides, improvement suggestions, and answers to your coffee-related questions.

## Features

- **Coffee Brewing Guides**: Get detailed instructions for various brewing methods (V60, French Press, Espresso, etc.)
- **Coffee Improvement Suggestions**: Input your current coffee parameters and get personalized suggestions
- **Coffee Knowledge Base**: Ask questions about coffee and get expert answers
- **Custom Knowledge Sources**: Add your own coffee knowledge through JSON files

## Adding Custom Knowledge Sources

You can extend Guaxinim's knowledge base by adding your own JSON files before running the application. Here's how:

1. Create a JSON file with your coffee knowledge in the following format:
```json
{
    "title": "Your Coffee Guide Title",
    "source": "https://your-source-url.com",
    "content": "Your detailed coffee knowledge content",
    "summary": "A brief summary of the content",
    "tags": ["coffee", "brewing", "guide"]
}
```

2. Place your JSON file in one of these directories:
   - `data/raw/pdf_processed/`: For processed PDF documents
   - `data/raw/youtube_processed/`: For processed YouTube transcripts

3. Run the embeddings creation script:
```bash
python scripts/create_embeddings.py
```

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

Guaxinim comes with several utility scripts to help you process different types of content:

### PDF Processing
```bash
# Process PDF files from data/raw/pdf directory
python scripts/process_pdfs.py
```
This script converts PDF files into JSON format with extracted text, summaries, and metadata.

### YouTube Content
```bash
# Crawl videos from any YouTube channel
python scripts/crawl_youtube_videos.py "https://www.youtube.com/@channelname"

# Process downloaded video transcripts
python scripts/process_transcripts.py
```
These scripts help you download and process YouTube video transcripts as knowledge sources. The crawler script requires a YouTube API key to be set in your `.env` file:

```
YOUTUBE_API_KEY=your_api_key_here
```

To get a YouTube API key:
1. Go to the [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Create credentials (API key)
5. Copy the API key and paste it in your `.env` file

### Embeddings Creation
```bash
# Create embeddings for all processed documents
python scripts/create_embeddings.py
```
Run this script after adding new content to update the search index.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
