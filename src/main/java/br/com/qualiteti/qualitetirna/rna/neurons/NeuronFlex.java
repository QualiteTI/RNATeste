package br.com.qualiteti.qualitetirna.rna.neurons;

import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;

public class NeuronFlex extends NeuronSuper {
	
	
	public NeuronFlex(EnActivationType activtion, ENCostFunctions cost) {
		super(EnNeuronType.flex);
		this.activationFunction = activtion;
		this.costFunction = cost;
		this.step = false;
	}

	
	@Override
	public double predict() {
		double out = aditive();
		double min = -1;
		double max = 1;
		switch(this.activationFunction) {
			case Linear:
				out = linear(out, false);
				min = 0;
				break;
			case Sigmoid:
				out = sigmoid(out, false);
				break;
			case Tanh:
				out = tanH(out, false);
				break;
			case ReLU:
				out = reLU(out, false);
				break;
			case LeakyReLU:
				out = leakReLU(out, false);
				break;
			case eLU:
				out = eLU(out, false);
				break;
		}
		
		return this.step?this.stepFunction(min, max, out):out;
	}
	
	
}
