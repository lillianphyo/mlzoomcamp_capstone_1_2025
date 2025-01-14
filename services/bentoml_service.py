# services/bentoml_service.py
import bentoml
from bentoml.io import JsonInput
from bentoml.io import KerasModelArtifact
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from config.config import config

@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([KerasModelArtifact('model')])
class EC2SpotForecastService(bentoml.BentoService):
    
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        data = parsed_json['data']
        data = np.array(data).reshape(-1, 1)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        data_scaled = data_scaled.reshape((1, 10, 1))
        prediction = self.artifacts.model.predict(data_scaled)
        prediction = scaler.inverse_transform(prediction)
        return prediction.tolist()

# Save the BentoML service
svc = EC2SpotForecastService()
svc.pack('model', load_model(config.MODEL_PATH))
saved_path = svc.save()