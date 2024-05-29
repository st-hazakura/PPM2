def evaluate_model(model, x_test, y_test):
    loss = model.evaluate(x_test, y_test, verbose=0)
    return loss
