import torch as T
class get_device:
	def __init__(self):
		super(get_device, self).__init__()
		self.device=T.device('cuda' if T.cuda.is_available() else 'cpu')


