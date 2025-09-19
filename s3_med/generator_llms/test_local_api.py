import unittest
# from generator_llms import local_api
# from generator_llms import local as local_api
from generator_llms import local_inst as local_api

class TestLocalAPI(unittest.TestCase):
    def test_generate_answer(self):
        print("\n=== Testing generate_answer with context ===")
        context = """The capital of France is Paris. The capital of Germany is Berlin.
Some irrelevant information: The sky is blue, water is wet, and 2+2=4.
The capital of Italy is Rome, but that's not relevant to this question.
The Eiffel Tower is in Paris, which is a famous landmark.
The capital of Spain is Madrid, but we're not asking about Spain."""
        question = "What is the capital of France?"
        print(f"Context: {context}")
        print(f"Question: {question}")
        answer = local_api.generate_answer(context, question)
        print(f"Answer: {answer}")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        self.assertIn("Paris", answer)

    def test_generate_answer_complex(self):
        print("\n=== Testing generate_answer with complex context ===")
        context = """In 2023, Paris hosted the Olympic Games. The city is known for its art and culture.
Berlin, on the other hand, is famous for its nightlife and history.
The capital of France is Paris, which is located in Europe.
The capital of Germany is Berlin, which is also in Europe.
Some random facts: The moon orbits Earth, the sun is hot, and computers use electricity.
The Louvre Museum is in Paris, which houses the Mona Lisa.
The Brandenburg Gate is in Berlin, which is a famous landmark.
The Seine River flows through Paris, while the Spree River flows through Berlin."""
        question = "What is the capital of France?"
        print(f"Context: {context}")
        print(f"Question: {question}")
        answer = local_api.generate_answer(context, question)
        print(f"Answer: {answer}")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        self.assertIn("Paris", answer)

    def test_generate_answer_contradictory(self):
        print("\n=== Testing generate_answer with contradictory context ===")
        context = """The capital of France is Paris. However, some people mistakenly believe it's Lyon.
The capital of Germany is Berlin, but there's a common misconception that it's Munich.
Paris is the largest city in France, while Berlin is the largest in Germany.
Some incorrect information: The capital of France is Marseille, which is false.
The Eiffel Tower is located in Paris, which is the correct capital.
The capital of Spain is Madrid, which is unrelated to this question."""
        question = "What is the capital of France?"
        print(f"Context: {context}")
        print(f"Question: {question}")
        answer = local_api.generate_answer(context, question)
        print(f"Answer: {answer}")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        self.assertIn("Paris", answer)

    def test_generate_answer_zero_shot(self):
        print("\n=== Testing generate_answer_zero_shot ===")
        question = "What is the capital of France?"
        print(f"Question: {question}")
        answer = local_api.generate_answer_zero_shot(question)
        print(f"Answer: {answer}")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        self.assertIn("Paris", answer)

    def test_call_llm(self):
        print("\n=== Testing call_llm ===")
        prompt = "What is 2+2?"
        print(f"Prompt: {prompt}")
        response = local_api.call_llm(prompt)
        print(f"Response: {response}")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_check_if_response_is_correct_llm(self):
        print("\n=== Testing check_if_response_is_correct_llm ===")
        response = "The answer is 4"
        gold_answers = ["4", "four"]
        print(f"Response to check: {response}")
        print(f"Gold answers: {gold_answers}")
        is_correct = local_api.check_if_response_is_correct_llm(response, gold_answers)
        print(f"Is correct: {is_correct}")
        self.assertIsInstance(is_correct, bool)

if __name__ == '__main__':
    unittest.main() 