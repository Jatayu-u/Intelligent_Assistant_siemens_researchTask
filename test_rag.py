import pytest
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from bert_score import score
from llm import query_rag
from langchain_groq import ChatGroq

# Configuration
FAISS_INDEX_PATH = "faiss_index"  # Path to the FAISS index for efficient similarity search
import os

# Fetch the API key for Hugging Face and Groq from environment variables
api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Prompt template used to evaluate the actual response against the expected response
EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

# Global lists for tracking metrics and responses
y_true = []  # Ground truth labels (1 for correct response)
y_pred = []  # Predicted labels (1 for correct, 0 for incorrect)
generated_responses = []  # List of model-generated responses
expected_responses = []  # List of expected responses for validation


def query_and_validate(question: str, expected_response: str):
    """
    Query the model with a given question and validate the response 
    against the expected response.

    Parameters:
        question (str): The input question to be queried.
        expected_response (str): The expected correct answer.

    You must respond True if the context of the response matches the expected_response. Check the entire response beforte deciding.    

    Returns:
        bool: True if the model response matches the expected response, False otherwise.
    """
    # Query the retrieval-augmented generation model
    response_text = query_rag(question)

    # Format the evaluation prompt with expected and actual responses
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    # Initialize the evaluation model using ChatGroq with specified parameters
    model = ChatGroq(
        temperature=0.7,
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        streaming=True,
        verbose=True,
    )
    evaluation_results = model.invoke(prompt)
    evaluation_results_str = evaluation_results.content.strip().lower()

    # Print prompt and model-generated response for transparency
    print("Prompt:", prompt)
    print("Response:", response_text)

    # Append responses to global lists for subsequent metrics calculation
    generated_responses.append(response_text)
    expected_responses.append(expected_response)

    # Evaluate if the model response matches the expected response
    if "true" in evaluation_results_str:
        print("\033[92m" + "✅ Correct" + "\033[0m")  # Green check mark for correct response
        y_true.append(1)
        y_pred.append(1)
        return True
    elif "false" in evaluation_results_str:
        print("\033[91m" + "❌ Incorrect" + "\033[0m")  # Red cross for incorrect response
        y_true.append(1)
        y_pred.append(0)
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )


@pytest.fixture(autouse=True)
def compute_metrics(request):
    """
    Pytest fixture to compute and print evaluation metrics after tests.

    Automatically runs after all tests to summarise the model's performance.
    """
    yield  # Allow test cases to execute before computing metrics
    if "pytestmark" in request.keywords:
        print("\n--- Metrics Summary ---")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")

        # Compute ROUGE scores for generated and expected responses
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = [
            scorer.score(ref, gen)
            for ref, gen in zip(expected_responses, generated_responses)
        ]
        print(f"ROUGE-1: {sum(r['rouge1'].fmeasure for r in rouge_scores) / len(rouge_scores):.4f}")
        print(f"ROUGE-2: {sum(r['rouge2'].fmeasure for r in rouge_scores) / len(rouge_scores):.4f}")
        print(f"ROUGE-L: {sum(r['rougeL'].fmeasure for r in rouge_scores) / len(rouge_scores):.4f}")

        # Compute BERTScore for generated and expected responses
        P, R, F1 = score(generated_responses, expected_responses, lang="en", verbose=True)
        print(f"BERTScore Precision: {P.mean().item():.4f}")
        print(f"BERTScore Recall: {R.mean().item():.4f}")
        print(f"BERTScore F1: {F1.mean().item():.4f}")

# Test cases
def test_check_art_parts_sections():
    assert query_and_validate(
        question="What is the total number of art, parts and sections in Indian Constitution currently?",
        expected_response="448 Art, 25 parts, 12 Schedules",
    )



def test_what_is_state():
    assert query_and_validate(
        question="What is a State according to Art 12?",
        expected_response="The state includes:- The Government and parliament of India The Government and legislature of each of the states. All local and other authorities: Within the territory of India Under the control of the Government of India All the fundamental rights are available against the state with a few exception.",
    )

# Test cases
def test_fundamental_rights_objective():
    assert query_and_validate(
        question="What is the aim of Fundamental Rights?",
        expected_response="To ensure certain elementary rights like right to life, liberty, freedom of speech and faith remain inviolable.",
    )


def test_fundamental_rights_magnum_carta():
    assert query_and_validate(
        question="What is Fundamental Right called in India?",
        expected_response="The Magna Carta of India",
    )


def test_exceptions_art15_3():
    assert query_and_validate(
        question="What does Article 15(3) permit?",
        expected_response="Special provision for women and children.",
    )


def test_exception_socially_backward_classes():
    assert query_and_validate(
        question="Which article allows provisions for socially and educationally backward classes?",
        expected_response="Article 15(4) and 15(5)",
    )


def test_equality_public_employment():
    assert query_and_validate(
        question="What does Article 16 ensure?",
        expected_response="Equality of opportunity in matters of public employment.",
    )


def test_article_16_exception_reservation():
    assert query_and_validate(
        question="What does Article 16 permit regarding backward classes?",
        expected_response="Reservation of posts for backward classes not adequately represented.",
    )


def test_double_jeopardy_protection():
    assert query_and_validate(
        question="What does the Double Jeopardy clause state?",
        expected_response="No person shall be prosecuted and punished for the same offence more than once.",
    )


def test_abolition_of_untouchability():
    assert query_and_validate(
        question="What does Article 17 abolish?",
        expected_response="Untouchability and its practice in any form.",
    )


def test_right_to_freedom():
    assert query_and_validate(
        question="Which article guarantees six freedoms?",
        expected_response="Article 19(1)",
    )


def test_prohibition_of_titles():
    assert query_and_validate(
        question="What does Article 18 prohibit?",
        expected_response="Conferring of titles except military or academic distinctions.",
    )


def test_freedom_of_speech_meaning():
    assert query_and_validate(
        question="What does freedom of speech include?",
        expected_response="The right to speak, express opinions, publish, and circulate information.",
    )


def test_freedom_of_silence():
    assert query_and_validate(
        question="Which case upheld the freedom of silence?",
        expected_response="Bijoy Emmanuel vs State of Kerala",
    )


def test_pre_censorship_invalid():
    assert query_and_validate(
        question="Which case declared pre-censorship invalid?",
        expected_response="Ramesh Thapper vs State of Madras",
    )


def test_right_to_know():
    assert query_and_validate(
        question="What is the principal behind the Right to Know?",
        expected_response="People's right to know about the functioning of the government.",
    )


def test_prabhu_datt_case():
    assert query_and_validate(
        question="What did the court decide in Prabhu Datt vs Union of India?",
        expected_response="The right to know news and information about government functioning.",
    )


def test_traffic_in_humans():
    assert query_and_validate(
        question="What does Article 23 prohibit?",
        expected_response="Traffic in human beings and forced labour.",
    )


def test_prohibition_of_child_labour():
    assert query_and_validate(
        question="What age restriction does Article 24 impose on child labour?",
        expected_response="Below the age of 14 years.",
    )


def test_freedom_of_religion():
    assert query_and_validate(
        question="What does Article 25 ensure?",
        expected_response="Freedom of conscience and right to profess, practice, and propagate religion.",
    )


def test_manage_religious_affairs():
    assert query_and_validate(
        question="What does Article 26 grant to religious denominations?",
        expected_response="Right to manage religious affairs, including property and institutions.",
    )


def test_taxation_for_religion():
    assert query_and_validate(
        question="What does Article 27 prohibit?",
        expected_response="Compulsion to pay taxes for the promotion or maintenance of any religion.",
    )


def test_freedom_from_religious_instruction():
    assert query_and_validate(
        question="What does Article 28 provide regarding religious instruction?",
        expected_response="No person can be forced to attend religious instruction in state-run institutions.",
    )


def test_ex_post_facto_law():
    assert query_and_validate(
        question="What does the Ex-post facto law state?",
        expected_response="No person shall be punished for an act not an offence at the time of its commission.",
    )


def test_reservation_in_promotions():
    assert query_and_validate(
        question="What reservation provision does Article 16 include?",
        expected_response="Reservation in promotion for Scheduled Castes and Scheduled Tribes.",
    )


def test_military_service_exception():
    assert query_and_validate(
        question="What is an exception to forced labour prohibition under Article 23?",
        expected_response="Compulsory service for public purposes like military service.",
    )


def test_flying_national_flag():
    assert query_and_validate(
        question="Which case recognised flying the national flag as a fundamental right?",
        expected_response="Union of India vs Naveen Jindal",
    )


def test_commercial_speech_validity():
    assert query_and_validate(
        question="Which case declared commercial speech as part of freedom of speech?",
        expected_response="Tata Press Ltd. vs MTNL",
    )


def test_right_to_press():
    assert query_and_validate(
        question="What is the freedom of the press under Article 19(1)?",
        expected_response="Right to publish and circulate information without pre-censorship.",
    )


def test_right_to_property():
    assert query_and_validate(
        question="What is the significance of Article 26 regarding property?",
        expected_response="Right to own, acquire, and manage property for religious purposes.",
    )


def test_religious_denomination_definition():
    assert query_and_validate(
        question="What conditions define a religious denomination?",
        expected_response="Body with common beliefs, organisation, and a distinctive name.",
    )


def test_untouchability_definition():
    assert query_and_validate(
        question="Is untouchability defined under the Constitution?",
        expected_response="No, but it refers to social disabilities imposed on certain castes.",
    )


def test_special_provision_educational_institutions():
    assert query_and_validate(
        question="What does Article 15(5) allow for backward classes?",
        expected_response="Special provisions for admission to private educational institutions.",
    )


def test_detention_exceeding_three_months():
    assert query_and_validate(
        question="What is the maximum period of detention without advisory board approval?",
        expected_response="Three months.",
    )


def test_ground_of_detention():
    assert query_and_validate(
        question="What must be communicated to a detained person?",
        expected_response="The grounds of detention.",
    )


def test_protection_of_culture():
    assert query_and_validate(
        question="What does Article 29 ensure?",
        expected_response="Protection of the interests of minorities and their culture.",
    )


def test_caste_discrimination_in_employment():
    assert query_and_validate(
        question="Does Article 16 allow discrimination in employment based on caste?",
        expected_response="No, caste-based discrimination is prohibited.",
    )


def test_reservation_adequately_represented():
    assert query_and_validate(
        question="What condition must be met for reservation of posts?",
        expected_response="Backward classes must be inadequately represented in services.",
    )


def test_social_service_exception():
    assert query_and_validate(
        question="What is an exception to Article 23 prohibition?",
        expected_response="Compulsory social service.",
    )


def test_compulsion_of_taxes_religion():
    assert query_and_validate(
        question="Can taxes be levied for religious promotion?",
        expected_response="No, Article 27 prohibits this.",
    )


def test_right_to_assembly():
    assert query_and_validate(
        question="What does Article 19(1)(b) guarantee?",
        expected_response="Freedom of Assembly.",
    )


def test_reasonable_restriction_on_speech():
    assert query_and_validate(
        question="What restrictions can be placed on freedom of speech?",
        expected_response="Security of State, public order, decency, defamation, and sovereignty.",
    )


def test_place_of_birth_discrimination():
    assert query_and_validate(
        question="Does Article 16 allow discrimination based on place of birth?",
        expected_response="No, place of birth discrimination is prohibited.",
    )


def test_fees_for_religious_institutions():
    assert query_and_validate(
        question="Can fees be charged for managing religious institutions?",
        expected_response="Yes, fees are allowed for secular administration.",
    )


def test_no_titles_conferred():
    assert query_and_validate(
        question="Can titles be conferred under Article 18?",
        expected_response="No, except military or academic distinctions.",
    )

@pytest.fixture(scope="session", autouse=True)
def calculate_metrics():
    """Calculate and display Precision, Recall, and F1-Score after all tests."""
    yield  # Wait for tests to complete
    if y_true and y_pred:  # Ensure lists are not empty
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print("\n--- Evaluation Metrics ---")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1-Score:  {f1:.2f}")
        print("--------------------------")
    else:
        print("No evaluation data available.")


if __name__ == "__main__":
    pytest.main()