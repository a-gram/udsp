

class System(object):
    """
    Abstract base class for all types of systems

    This class defines the interface for a system with n inputs
    and m outputs (a MIMO system, loosely speaking).

    Attributes
    ----------
    inputs: list[Signal]
        An array of input signals [x1,...,xn]
    outputs: list[Signal]
        (read-only)
        An array of output signals [y1,...,ym]

    """
    def __init__(self):

        super().__init__()
        self.inputs = []
        self._outputs = []

    def process(self, inputs=None):
        """
        Process the input and initiate generation of the output

        This method shall be called to initiate the processing of the
        input signals and the generation of the output signals. In its
        basic implementation it simply invokes _sysop(), but derived
        classes may override it to implement additional functionality
        (such as preprocessing of the inputs, etc.) and then call this
        base method to perform the processing.

        Parameters
        ----------
        inputs: list[Signal], Signal, optional
            An array of input signals to be processed. If not given, the
            input is taken from the instance attribute. If given, it will
            be assigned to the instance attribute. It also accepts a
            single Signal instance, which is internally wrapped in a list.

        Returns
        -------
        list[Signals]
            An array of output signals

        """
        inputs = inputs if type(inputs) is list else [inputs]
        self.inputs = inputs or self.inputs

        if not self.inputs:
            raise ValueError(
                "No input signals"
            )

        self._outputs = self._sysop()
        return self._outputs

    def _sysop(self):
        """
        The system operator

        This abstract method represents the operation that combines the
        inputs with the system's model (e.g. the impulse response for
        linear systems) in order to generate the output.
        Generally, given n input signals this function produces m output
        signals according to the relation [y1,...,ym] = S([x1,...,xn], G),
        where G represents the system's model.

        Returns
        -------
        list[Signal]
            An array of output signals

        """
        raise NotImplementedError

    @property
    def outputs(self):
        return self._outputs
