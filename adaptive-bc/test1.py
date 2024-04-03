import unittest
from run import kwparams, run_model
from unittest.mock import patch
from numpy.random import SeedSequence

class TestRunModel(unittest.TestCase):
    def setUp(self):
        # 为测试设置固定的参数
        self.N = 10
        self.C = 2
        self.beta = 0.5
        self.trial = 1
        self.K = 3
        self.seed_sequence = SeedSequence(12345)
        self.model_params = kwparams(self.N, self.C, self.beta, self.trial, self.K)
        self.filename = 'test_model'

    @patch('run.Model')
    def test_run_model(self, MockModel):
        # 模拟Model的行为
        mock_model_instance = MockModel.return_value
        mock_model_instance.run.return_value = None
        mock_model_instance.save_model.return_value = None
        mock_model_instance.beta = self.beta
        
        # 运行模型
        run_model(self.seed_sequence, self.model_params, self.filename)
        
        # 检查run和save_model是否被调用
        mock_model_instance.run.assert_called_once_with(test=True)
        mock_model_instance.save_model.assert_called_once_with(f'data/{self.filename}.pbz2')

        # 如果beta不为1，检查输出（这里我们不检查stdout，只确保没有报错）
        if self.beta != 1:
            pass  # 这里我们可以添加更复杂的逻辑来检查输出，但通常不需要

if __name__ == '__main__':
    unittest.main()