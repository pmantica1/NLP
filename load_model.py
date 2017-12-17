from database import TransferLearningDatabase
from cnn import CNN
from nn_utils import test_auc
import torch


def load_cnn_encoder(filepath, feature_vector_dimensions, questions_vector_dimensions, kernel_size):
	encoder = CNN(feature_vector_dimensions, questions_vector_dimensions, kernel_size)
	encoder.load_state_dict(torch.load(filepath))
	return encoder


if __name__ == "__main__":
	database = TransferLearningDatabase()
	test_set = database.get_testing_set()

	filepath = "cnn_encoder.pt"
	feature_vector_dimensions = 300
	questions_vector_dimensions = 500
	kernel_size = 3

	encoder = CNN(feature_vector_dimensions, questions_vector_dimensions, kernel_size)
	encoder.load_state_dict(torch.load(filepath))

	print test_auc(encoder, test_set)