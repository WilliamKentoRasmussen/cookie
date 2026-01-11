from src.my_project.model import MyAwesomeModel
import torch 
model = MyAwesomeModel()
import pytest



def test_model2():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10), "Expected to have 10 different labels"

'''
def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
        model(torch.randn(1,1,28,29))
        '''
@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)



'''
@pytest.mark.parametrize("network_size", [10, 100, 1000])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
class MyTestClass:
    @pytest.mark.parametrize("network_type", ["alexnet", "squeezenet", "vgg", "resnet"])
    @pytest.mark.parametrize("precision", [torch.half, torch.float, torch.double])
    def test_network1(self, network_size, device, network_type, precision):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("Test requires cuda")
        model = MyModelClass(network_size, network_type).to(device=device, dtype=precision)
        ...

    @pytest.mark.parametrize("add_dropout", [True, False])
    def test_network2(self, network_size, device, add_dropout):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("Test requires cuda")
        model = MyModelClass2(network_size, add_dropout).to(device)
        ...

'''