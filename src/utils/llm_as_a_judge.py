
import json
import re
from typing import Tuple

from src.utils.llm import generate_text, split_thinking_output
from src.utils.prompts import LLM_CONTEST_SYSTEM_PROMPT, LLM_CONTEST_USER_PROMPT, LLM_OF_JUDGE_SYSTEM_PROMPT, LLM_OF_JUDGE_USER_PROMPT


def get_llm_judge_prediction(model, tokenizer, ref_1: str, ref_2: str, ref_3: str, target_abstract: str) -> Tuple[int, str]:
    """
    Baut den Prompt für den LLM-Judge, generiert die Bewertung und 
    extrahiert Score sowie Reasoning aus der JSON-Antwort.
    """
    prompt = LLM_OF_JUDGE_USER_PROMPT.format(
        example_1=ref_1,
        example_2=ref_2,
        example_3=ref_3,
        gen_abstract=target_abstract
    )

    response_text = generate_text(
        model,
        tokenizer,
        LLM_OF_JUDGE_SYSTEM_PROMPT,
        prompt,
        max_tokens=1024,
        do_sample=False
    )

    print(f"\nRaw LLM JUDGE Response: {response_text}\n")

    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)

            score = int(result.get("score", -1))
            reasoning = result.get(
                "reasoning", "Kein Reasoning im JSON gefunden.")

            return score, reasoning
        else:
            # Fallback falls das LLM gar kein JSON generiert hat
            return -1, f"Parsing Error: Kein JSON-Block gefunden. Raw: {response_text}"

    except json.JSONDecodeError:
        return -1, "Parsing Error: JSON war syntaktisch inkorrekt (z.B. fehlende Anführungszeichen)."
    except ValueError:
        return -1, "Parsing Error: Der generierte Score war keine gültige Zahl."
    except Exception as e:
        return -1, f"Unerwarteter Fehler: {str(e)}"


def get_llm_judge_1v1(model, tokenizer, abstract_a: str, abstract_b: str) -> Tuple[str, str]:
    """
    Baut den Prompt für den LLM-Judge, generiert die Bewertung und 
    extrahiert Score sowie Reasoning aus der JSON-Antwort.
    """
    prompt = LLM_CONTEST_USER_PROMPT.format(
        abstract_a=abstract_a,
        abstract_b=abstract_b
    )

    response_text = generate_text(
        model,
        tokenizer,
        LLM_CONTEST_SYSTEM_PROMPT,
        prompt,
        max_tokens=1536,
        do_sample=False,
        temperature=0.3
    )

    response, _ = split_thinking_output(response_text)

    print(f"\nRaw LLM JUDGE 1v1 Response: {response}\n")

    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)

            # Lese das Reasoning aus
            reasoning = result.get(
                "reasoning", "Kein Reasoning im JSON gefunden.")

            # Hole den Gewinner, entferne Leerzeichen und mache alles groß (für Robustheit)
            winner = str(result.get("winner", "")).strip().upper()

            # Validiere, ob das Modell sich an die Vorgaben gehalten hat
            if winner not in ["A", "B"]:
                return "ERROR", f"Validation Error: Ungültiger Gewinner generiert ('{winner}'). Raw: {json_str}"

            return winner, reasoning
        else:
            return "ERROR", f"Parsing Error: Kein JSON-Block gefunden. Raw: {response}"

    except json.JSONDecodeError:
        return "ERROR", "Parsing Error: JSON war syntaktisch inkorrekt (z.B. fehlende Anführungszeichen)."
    except Exception as e:
        return "ERROR", f"Unerwarteter Fehler: {str(e)}"
