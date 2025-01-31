# Guaxinim - Your Personal Coffee Assistant 🦝☕

Guaxinim is an AI-powered coffee assistant that helps you brew the perfect cup of coffee. It provides brewing guides, improvement suggestions, and answers to your coffee-related questions.

## Features

- **Coffee Brewing Guides**: Get detailed instructions for various brewing methods (V60, French Press, Espresso, etc.)
- **Coffee Improvement Suggestions**: Input your current coffee parameters and get personalized suggestions
- **Coffee Knowledge Base**: Ask questions about coffee and get expert answers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lfomendes/guaxinim-barista.git
cd guaxinim-barista
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

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

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
