"""
Guaxinim - Your Personal Coffee Barista Assistant
This module implements a Streamlit-based web interface for a coffee preparation assistant.
The app provides three main functionalities:
1. Guide users in making the perfect coffee
2. Help users improve their current coffee preparation
3. Answer coffee-related questions using AI
"""

import streamlit as st
from coffee_data import CoffeePreparationData
from guaxinim_bot import GuaxinimBot


# Initialize the bot
guaxinim_bot = GuaxinimBot()

# Define brewing methods globally
BREWING_METHODS = [
    "V60",
    "French Press",
    "Espresso",
    "Aeropress",
    "Cold Brew",
    "Moka Pot",
    "Chemex",
]


def get_coffee_preparation_data(show_all_fields: bool = True) -> CoffeePreparationData:
    """
    Collects coffee preparation parameters from the user interface.

    Args:
        show_all_fields (bool): If True, shows all available fields. If False, shows only
            mandatory fields.

    Returns:
        CoffeePreparationData: A dataclass containing all the coffee preparation parameters
    """

    issue = st.selectbox(
        "Issue encountered (required):",
        ["Too acidic", "Too bitter", "Seems under-extracted", "Other"],
    )

    amount_of_coffee = st.number_input(
        "Amount of coffee in grams (required):",
        min_value=0.0,
        max_value=500.0,
        step=1.0,
    )

    amount_of_water = st.number_input(
        "Amount of water in ml (required):",
        min_value=0.0,
        max_value=2000.0,
        step=1.0
    )

    if show_all_fields:
        st.markdown("---")
        st.markdown("#### Optional Parameters")
        st.markdown(
            "The following parameters are optional but will help provide more accurate "
            "suggestions if provided:"
        )

        type_of_bean = st.text_input("Type of bean:")
        type_of_bean = type_of_bean if type_of_bean else None

        total_extraction_time = st.number_input(
            "Total extraction time (seconds):",
            min_value=0,
            max_value=600,
            step=1,
            value=0
        )
        total_extraction_time = total_extraction_time if total_extraction_time > 0 else None

        water_temperature = st.number_input(
            "Water temperature (°C):",
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            value=0.0
        )
        water_temperature = water_temperature if water_temperature > 0 else None

        grinder_granularity = st.selectbox(
            "Grinder granularity:",
            ["", "Fine", "Medium-Fine", "Medium", "Medium-Coarse", "Coarse"],
        )
        grinder_granularity = grinder_granularity if grinder_granularity else None

        bloom_time = st.number_input(
            "Bloom time (seconds):",
            min_value=0,
            max_value=120,
            step=1,
            value=0
        )
        bloom_time = bloom_time if bloom_time > 0 else None

        number_of_pours = st.number_input(
            "Number of pours:",
            min_value=0,
            max_value=10,
            step=1,
            value=0
        )
        number_of_pours = number_of_pours if number_of_pours > 0 else None

        amount_in_each_pour = st.number_input(
            "Amount in each pour (ml):",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            value=0.0
        )
        amount_in_each_pour = amount_in_each_pour if amount_in_each_pour > 0 else None

        notes = st.text_area("Additional notes:")
        notes = notes if notes else None

        return CoffeePreparationData(
            issue_encountered=issue,
            amount_of_coffee=amount_of_coffee,
            amount_of_water=amount_of_water,
            type_of_bean=type_of_bean,
            total_extraction_time=total_extraction_time,
            water_temperature=water_temperature,
            grinder_granularity=grinder_granularity,
            bloom_time=bloom_time,
            number_of_pours=number_of_pours,
            amount_in_each_pour=amount_in_each_pour,
            notes=notes,
        )

    return CoffeePreparationData(
        issue_encountered=issue,
        amount_of_coffee=amount_of_coffee,
        amount_of_water=amount_of_water,
    )


def learn_coffee_making():
    """
    Displays the page for learning how to make perfect coffee.
    Provides brewing method selection and displays best practices for the selected method.
    Uses OpenAI to generate detailed brewing instructions.
    """

    st.header("Learn to Make Perfect Coffee")
    method = st.selectbox("Select your brewing method:", BREWING_METHODS)

    if st.button("Get Brewing Guide"):
        with st.spinner("Generating your personalized brewing guide..."):
            guide = guaxinim_bot.get_coffee_guide(method)
            st.markdown(guide)


def improve_coffee():
    """
    Displays the page for improving existing coffee preparation.
    Collects current coffee preparation parameters and provides AI-powered suggestions
    for improvement using the GuaxinimBot.
    """

    st.header("Improve Your Coffee")
    st.write(
        "Tell me about your current coffee preparation, and I'll help you improve it."
    )

    coffee_data = get_coffee_preparation_data()

    if st.button("Get Improvement Suggestions"):
        with st.spinner("Analyzing your coffee parameters..."):
            suggestions = guaxinim_bot.improve_coffee(coffee_data)
            st.markdown(suggestions)


def learn_about_coffee():
    """
    Displays the coffee knowledge page where users can ask coffee-related questions.
    Provides pre-defined question suggestions and allows custom questions.
    Uses GuaxinimBot to provide AI-powered answers about coffee.
    """

    st.header("Learn About Coffee")
    st.write("Choose a question or ask your own!")

    # Define suggested questions
    questions = [
        "What is coffee bloom?",
        "How does grind size affect coffee extraction?",
        "What's the difference between Arabica and Robusta?",
        "How should I store coffee beans?",
    ]

    # Create columns for better button layout
    cols = st.columns(2)
    
    # Create buttons for suggested questions
    selected_question = None
    for i, question in enumerate(questions):
        col_idx = i % 2
        if cols[col_idx].button(question):
            selected_question = question

    # Add custom question button and input
    st.write("")  # Add some spacing
    
    # Use a form for custom questions
    if st.button("Ask your own question ", type="primary"):
        st.session_state.show_custom = True
    
    if 'show_custom' in st.session_state and st.session_state.show_custom:
        with st.form(key='custom_question_form'):
            custom_question = st.text_input("What would you like to know about coffee?")
            submit_button = st.form_submit_button("Get Answer")
            if submit_button and custom_question.strip():
                selected_question = custom_question

    # Display answer if a question is selected
    if selected_question and selected_question.strip():
        with st.spinner("Getting answer..."):
            answer = guaxinim_bot.ask_guaxinim(selected_question)
            st.write("### Answer")
            st.write(answer)


def main():
    """
    Main application entry point.
    Sets up the Streamlit page configuration, applies custom styling,
    and initializes the main application interface.
    """

    st.set_page_config(
        page_title="Guaxinim - Coffee Assistant", page_icon=" ", layout="wide"
    )

    # Create three columns with the middle one being 600px wide
    left_col, center_col, right_col = st.columns([1, 2, 1])

    with center_col:
        st.title("Guaxinim, Your Personal Barista")
        _, img_col, _ = st.columns(3)
        with img_col:
            st.image("img/guaxinim.png", width=250)

        option = st.radio(
            "Choose an option:",
            [
                "Learn to make perfect coffee",
                "Improve my current coffee",
                "Learn about coffee",
            ],
        )

        if option == "Learn to make perfect coffee":
            learn_coffee_making()
        elif option == "Improve my current coffee":
            improve_coffee()
        else:
            learn_about_coffee()


if __name__ == "__main__":
    main()
