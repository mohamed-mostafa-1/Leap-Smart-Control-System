from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def hyperparameter_tuning(X, y):
    def create_model(lstm_units=64, dropout_rate=0.2):
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(dropout_rate),
            LSTM(lstm_units),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dense(len(set(y)), activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=32, verbose=0)
    param_grid = {'lstm_units': [32, 64, 128], 'dropout_rate': [0.2, 0.3, 0.4]}
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X, y)
    
    return grid_result.best_params_
