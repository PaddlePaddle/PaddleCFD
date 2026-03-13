import os
import paddle


def save_model(model, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    paddle.save(model.state_dict(), path)

    print("Model saved to:", path)


def load_model(model, path):

    state_dict = paddle.load(path)

    model.set_state_dict(state_dict)

    print("Model loaded:", path)