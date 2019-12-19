from models.frn import FRN, FRN3d, TLU3d,FRNTLU3d
import torch

def test_frn():
    data = torch.ones((2,3,10,10))
    frn = FRN(3)
    output = frn(data)
    assert output.shape == (2, 3, 10, 10)

def test_frn3d():
    data = torch.ones((2, 3, 5, 10, 10))
    frn = FRN3d(3)
    output = frn(data)
    assert output.shape == (2, 3, 5, 10, 10)

def test_tlu3d():
    data = torch.ones((2, 3, 5, 10, 10))
    tlu = TLU3d(3)
    output = tlu(data)
    assert output.shape == (2, 3, 5, 10, 10)

def test_frntlu3d():
    data = torch.ones((2, 3, 5, 10, 10))
    frntlu = FRNTLU3d(3)
    output = frntlu(data)
    assert output.shape == (2, 3, 5, 10, 10)