from mango.messages.codecs import json_serializable


@json_serializable
class AggregatedSolutionMessage:
    """
        Message to send an aggregated Solution
        """

    def __init__(self, aggregated_solution, aggregated_flexibility, dates):
        self._aggregated_solution = aggregated_solution
        self._aggregated_flexibility = aggregated_flexibility
        self._dates = dates

    @property
    def aggregated_solution(self):
        return self._aggregated_solution

    @property
    def aggregated_flexibility(self):
        return self._aggregated_flexibility

    @property
    def dates(self):
        return self._dates


@json_serializable
class RedispatchFlexibilityRequest:
    """
    Message to ask agents for redispatch flexibility
    """

    def __init__(self, dates, obligations):
        self._dates = dates
        self._obligations = obligations

    @property
    def dates(self):
        return self._dates

    @property
    def obligations(self):
        return self._obligations


@json_serializable
class RedispatchFlexibilityReply:
    """
    Message for agents to send redispatch flexibility
    """

    def __init__(self, dates, flexibility):
        self._dates = dates
        self._flexibility = flexibility

    @property
    def dates(self):
        return self._dates

    @property
    def flexibility(self):
        return self._flexibility


@json_serializable
class CallForAdaption:
    """
        Message to send for call for adaption in scheduled power values
        """

    def __init__(self, obligation_date, obligation_value):
        self._obligation_date = obligation_date
        self._obligation_value = obligation_value

    @property
    def obligation_date(self):
        return self._obligation_date

    @property
    def obligation_value(self):
        return self._obligation_value
