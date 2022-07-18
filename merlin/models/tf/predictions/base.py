from tensorflow.keras.layers import Layer

from merlin.models.tf.core.transformations import LogitsTemperatureScaler


class PredictionBlock(Layer):
	def __init__(
		self,
		prediction,
		default_loss,
		default_metrics,
		target=None,
		pre=None,
		post=None,
		logits_temperature=1.0
	):
		self.prediction = prediction
		self.default_loss = default_loss
		self.default_metrics = default_metrics
		self.target = target
		self.pre = pre
		self.post = post
		self.logits_temperature = logits_temperature
		if logits_temperature != 1.0:
			self.logits_scaler = LogitsTemperatureScaler(logits_temperature)

	def call(self, inputs, context):
		return self.prediction(inputs, context)

	def __call__(self, inputs, *args, **kwargs):
		# call pre
		if self.pre:
			inputs = self.pre(inputs, *args, **kwargs)

		# super call
		outputs = super().__call__(inputs, *args, **kwargs)

		if self.post:
			outputs = self.post(outputs, *args, **kwargs)

		if getattr(self, "logits_scaler", None):
			outputs = self.logits_scaler(outputs)

		return outputs

