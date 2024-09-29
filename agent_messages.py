from typing import Dict
from typing import List, Tuple

from arrow import arrow

from mango.messages.codecs import json_serializable
from mango_library.negotiation.multiobjective_cohda.data_classes import SolutionPoint


@json_serializable
class AgentAddress:
    """
    The address of an agent
    """

    def __init__(self, host, port, aid, agent_type=None):
        self.host = host
        self.port = port
        self.aid = aid
        if agent_type is None:
            agent_type = ""
        self.agent_type = agent_type  # neighborhood, grid, unit, observer

    def __eq__(self, other) -> bool:
        if self.host == other.host \
                and self.port == other.port \
                and self.aid == other.aid \
                and self.agent_type == other.agent_type:
            return True
        return False

    def __hash__(self):
        return hash((self.host, self.port, self.aid, self.agent_type))


@json_serializable
class CohdaSolution:
    def __init__(self):
        self.solution_points: List[SolutionPoint] = []
        self.corr_negotiation_id = None
        # dict mit negotiation_id:SolutionPoint


@json_serializable
class NewStep:

    def __init__(self, start_date):
        self.start_date = start_date


@json_serializable
class CallForAdaption:

    def __init__(self, obligations, start_date):
        self.obligations = obligations  # obligation is list with dates and given values for these dates
        self.start_date = start_date  # start date for complete negotiation interval


@json_serializable
class SendInvitations:
    def __init__(self, start_date):
        self.start_date = start_date


@json_serializable
class AggregatedRedispatchFlex:

    def __init__(self, aggr_redispatch_flex, aggr_forecast):
        # TOOD uid
        self.aggr_redispatch_flex = aggr_redispatch_flex
        self.aggr_forecast = aggr_forecast


@json_serializable
class RedispatchFlexibilityRequest:
    """Message for inviting an agent to a coalition.
    """

    def __init__(self, start_date: str, corr_negotiation_id: int, length=None):
        self._start_date = start_date
        self._length = length
        self._corr_negotiation_id = corr_negotiation_id

    @property
    def start_date(self) -> arrow:
        """Return start date of the flexibility

        :return: start date of the flexibility
        """
        return self._start_date

    @property
    def length(self) -> arrow:
        """Return length of the flexibility

        :return: length of the flexibility
        """
        return self._length

    @property
    def corr_negotiation_id(self) -> int:
        """Return negotiation_id of the request

        :return: negotiation_id of the request
        """
        return self._corr_negotiation_id


@json_serializable
class RedispatchFlexibilityReply:
    """
    Flexibility of agent: interval of how much power the agent can offer
    """

    def __init__(self, start_date: str, negotiation_id, flexibility=None):
        if flexibility is None:
            flexibility = []
        self.flexibility: List[Tuple[str, float, float]] = flexibility
        self.start_date = start_date
        self.negotiation_id = negotiation_id


@json_serializable
class UpdateStateMessage:
    """
    MosaikAgent -> UnitAgent
    The state is the power value from mosaik
    """

    def __init__(self, state):
        self.state: Dict = state


@json_serializable
class RequestSetPointMsg:
    """
    MosaikAgent -> UnitAgent
    Request the current set point for the units
    """

    def __init__(self, unit_id):
        self.unit_id: str = unit_id


@json_serializable
class UnitSetPointMsg:
    """
    UnitAgent -> MosaikAgent
    Contains the current set point for the units
    """

    def __init__(self, set_point):
        self.set_point: float = set_point


@json_serializable
class AggregatedSetPointMsg:
    """
    Aggregator -> MosaikAgent
    Contains the current set point for the units
    """

    def __init__(self, set_point):
        self.set_point: float = set_point


@json_serializable
class AgentStepDoneMsg:
    """
    GridAgent -> MosaikAgent
    """

    def __init__(self, start_date):
        self.start_date = start_date


@json_serializable
class MosaikTrigger:
    def __init__(self, start_date):
        self.start_date = start_date
