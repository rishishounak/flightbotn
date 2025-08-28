import requests
from typing import Dict, Any, List
from datetime import datetime
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

# --- CONFIGURATION ---
AVIATIONSTACK_KEY = ""
OPENAI_API_KEY = ""
city_to_iata = {"Delhi": "DEL", "Mumbai": "BOM", "Bangalore": "BLR"}

# --- STATE PERSISTENCE & CACHE ---
class WorkflowState:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.context: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.clarification_needed: bool = False
        self.clarifying_question: str = ""

cached_flights: Dict[str, List[Dict[str, Any]]] = {}

# ---------------- NODES ----------------
def parse_request_text(state: WorkflowState):
    text = state.context["raw_text"].lower()
    cities = [city for city in city_to_iata.keys() if city.lower() in text or city_to_iata[city].lower() in text]
    state.context["cities"] = cities
    return state

def extract_entities(state: WorkflowState):
    text = state.context["raw_text"].lower()
    if "morning" in text:
        state.context["time"] = "morning"
    elif "afternoon" in text:
        state.context["time"] = "afternoon"
    elif "night" in text:
        state.context["time"] = "night"
    else:
        state.context["time"] = None
    return state

def solution_evaluation(state: WorkflowState):
    missing_fields = []
    if not state.context.get("cities"):
        missing_fields.append("departure city")
    if not state.context.get("time"):
        missing_fields.append("departure time (morning/afternoon/night)")
    if missing_fields:
        state.clarification_needed = True
        state.clarifying_question = f"Could you please specify {', '.join(missing_fields)}?"
    else:
        state.clarification_needed = False
    return state

def normalize_fields(state: WorkflowState):
    iatas = []
    for city in state.context.get("cities", []):
        iata = city_to_iata.get(city)
        if iata:
            iatas.append(iata)
    state.context["iatas"] = iatas
    return state

def store_answer(state: WorkflowState, key: str):
    cached_flights[key] = state.results.get("flights", [])
    return state

def retrieve_from_cache(state: WorkflowState, key: str):
    if key in cached_flights:
        state.results["flights"] = cached_flights[key]
    return state

# --- Filter flights by morning/afternoon/night ---
def filter_flights_by_time(flights: List[Dict[str, Any]], time_bucket: str) -> List[Dict[str, Any]]:
    filtered = []
    for f in flights:
        sched = f.get("departure", {}).get("scheduled")
        if not sched:
            continue
        dt = datetime.fromisoformat(sched.replace("Z", "+00:00"))
        hour = dt.hour
        if time_bucket == "morning" and 5 <= hour < 12:
            filtered.append(f)
        elif time_bucket == "afternoon" and 12 <= hour < 17:
            filtered.append(f)
        elif time_bucket == "night" and (hour >= 17 or hour < 5):
            filtered.append(f)
    return filtered

def execute_api_calls(state: WorkflowState):
    all_flights_combined = []

    for iata in state.context.get("iatas", []):
        cache_key = f"{iata}_{state.context.get('time', 'any')}"
        state = retrieve_from_cache(state, cache_key)
        flights = state.results.get("flights", [])
        if flights:
            all_flights_combined.extend(flights)
            continue

        max_pages = 2
        page = 1
        flights_data = []

        while page <= max_pages:
            url = f"http://api.aviationstack.com/v1/flights?access_key={AVIATIONSTACK_KEY}&dep_iata={iata}&limit=100&offset={(page-1)*100}"
            try:
                res = requests.get(url, timeout=10)
                res.raise_for_status()
                data = res.json().get("data", [])
            except requests.RequestException as e:
                print(f"AviationStack request failed for {iata} page {page}: {e}")
                break

            if state.context.get("time"):
                data = filter_flights_by_time(data, state.context["time"])
            if not data:
                break

            flights_data.extend(data)
            page += 1

        all_flights_combined.extend(flights_data)
        state.results["flights"] = flights_data
        state = store_answer(state, cache_key)

    state.results["flights"] = all_flights_combined
    return state

def complete_payload(state: WorkflowState):
    flights_data = state.results.get("flights", [])
    if not flights_data:
        return {"result": "No flights found."}

    merged_output = {}
    for f in flights_data:
        origin = f["departure"]["airport"]
        if origin not in merged_output:
            merged_output[origin] = []
        merged_output[origin].append({
            "airline": f["airline"]["name"],
            "flight_iata": f["flight"]["iata"],
            "departure": f["departure"]["scheduled"],
            "arrival": f["arrival"]["airport"]
        })
    return {"result": merged_output}

# ---------------- CHATBOT INTERFACE ----------------
def chatbot():
    print("🛫 FlightBot Chat (type 'exit' to quit)")
    state: WorkflowState = None

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        if state is None:
            payload = {"q": user_input}
            state = WorkflowState(payload)
            state.context["raw_text"] = user_input
        else:
            # Merge clarification input into raw_text
            state.context["raw_text"] += " " + user_input

        # Run deterministic nodes
        for node in [parse_request_text, extract_entities, solution_evaluation]:
            state = node(state)

        if state.clarification_needed:
            print("\nFlightBot:", state.clarifying_question)
            continue  # wait for next user input

        # All required info present → run remaining nodes
        for node in [normalize_fields, execute_api_calls]:
            state = node(state)

        result = complete_payload(state)
        print("\nFlightBot:", result["result"])

        # Reset state for next independent query
        state = None

if __name__ == "__main__":
    chatbot()
