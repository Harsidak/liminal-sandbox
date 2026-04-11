# Mock CCXT module to bypass FinRL dependency crash 
# without breaking the environment

class MockObject:
    def __getattr__(self, name):
        return MockObject()
    def __call__(self, *args, **kwargs):
        return MockObject()

import sys
sys.modules[__name__] = MockObject()
