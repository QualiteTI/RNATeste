package br.com.qualiteti.qualitetirna.rna.definition;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;

public class NeuronDefinition implements Serializable {
	private static final long serialVersionUID = 1L;
	public EnNeuronType type;
	public EnActivationType activationFunction;
	public ENCostFunctions costFunction;
	public double bias = 0;
	public double alfaReLU = 0;
	public double learningRateW = 0.1;
	public double learningRateB = 0.1;
	public boolean step = false;
	public List<Double> weights = new ArrayList<>();
}
