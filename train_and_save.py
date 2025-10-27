from model_utils import Network, load_mnist_data, save_model
import os

def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    training_data, test_data = load_mnist_data()
    net = Network([784, 64, 64, 10])
    # Small quick training so the demo is responsive. Increase epochs for better accuracy.
    net.sgd(training_data, epochs=30, mini_batch_size=20, eta=3.0, test_data=test_data)
    save_model(net, "model.pkl")
    print("Saved model to model.pkl")


if __name__ == "__main__":
    main()
