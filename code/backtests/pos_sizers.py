import backtrader as bt
import math

class maxRiskSizer(bt.Sizer):
    '''
    Returns the number of shares rounded down that can be purchased for the
    max risk tolerance
    '''
    params = (('risk', 0.05),)

    def __init__(self):
        if self.p.risk > 1 or self.p.risk < 0:
            raise ValueError('The risk parameter is a percentage which must be'
                'entered as a float. e.g. 0.5')

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy == True:
            size = math.floor((cash * self.p.risk) / data[0])
        else:
            size = math.floor((cash * self.p.risk) / data[0]) * -1
        return size

    
class LongOnly(bt.Sizer):
    params = (('risk', 1),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return math.floor((cash * self.p.risk) / data[0])

        # Sell situation
        position = self.strategy.getposition(data)
        if not position.size:
            return 0  # do not sell if nothing is open

        return position.size
    

class FixedReverser(bt.Sizer):
    params = (('stake', 1),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data)
        size = self.p.stake * (1 + (position.size != 0))
        return size