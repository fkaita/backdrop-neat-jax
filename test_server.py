import unittest
import json
import logging
from server import app

# Configure logging to export errors
logging.basicConfig(
    filename="test_errors.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR
)


class TestNeatAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def log_error(self, response, test_name):
        """Logs the error details if a test fails."""
        if response.status_code != 200:
            error_message = {
                "test_name": test_name,
                "status_code": response.status_code,
                "response_data": response.data.decode("utf-8"),
            }
            logging.error(json.dumps(error_message, indent=4))

    def test_index(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)

    def test_init_network(self):
        payload = {
            "nInput": 2,
            "nOutput": 1,
            "initConfig": "one",
            "activations": "minimal"
        }
        response = self.app.post("/init", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "initialized")
        self.assertEqual(data["nInput"], 2)
        self.assertEqual(data["nOutput"], 1)

    def test_network_api(self):
        response = self.app.get("/network")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("nInput", data)
        self.assertIn("nOutput", data)
        self.assertIn("nodes", data)
        self.assertIn("connections", data)
        self.assertIn("generation", data)

    def test_create_trainer(self):
        payload = {
            "mutation_rate": 0.1,
            "mutation_size": 0.2,
            "extinction_rate": 0.3
        }
        response = self.app.post("/create_trainer", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "trainer created")

    def test_evolve(self):
        self.test_create_trainer()  # Ensure a trainer is created
        payload = {"mutateWeightsOnly": True}
        response = self.app.post("/evolve", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "evolution step completed")

    def test_best_genome(self):
        self.test_create_trainer()  # Ensure a trainer is created
        response = self.app.get("/best_genome")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("best_genome", data)

    def test_apply_fitness(self):
        self.test_create_trainer()  # Ensure a trainer is created
        payload = {
            "fitness_config": "lambda genome: genome.fitness",
            "cluster_mode": True
        }
        response = self.app.post("/apply_fitness", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "fitness function applied")

    def test_forward(self):
        self.test_create_trainer()  # Ensure a trainer is created
        payload = {
            "input_values": [0.5, 0.3]
        }
        response = self.app.post("/forward", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("outputs", data)
        self.assertIsInstance(data["outputs"], list)

    def test_backward(self):
        self.test_create_trainer()  # Ensure a trainer is created
        payload = {
            "input_values": [0.5, 0.3],
            "target": [1.0],
            "nCycles": 10,
            "learnRate": 0.01
        }
        response = self.app.post("/backward", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("loss", data)
        self.assertIn("weights", data)
        self.assertIsInstance(data["weights"], list)

if __name__ == "__main__":
    unittest.main()
