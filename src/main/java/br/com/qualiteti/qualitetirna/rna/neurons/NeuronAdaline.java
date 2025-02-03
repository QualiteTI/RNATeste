package br.com.qualiteti.qualitetirna.rna.neurons;

import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;
import br.com.qualiteti.qualitetirna.common.types.LearnResult;

public class NeuronAdaline extends NeuronSuper {

	public NeuronAdaline() {
		super(EnNeuronType.adalaine);
		this.activationFunction = EnActivationType.Linear;
		this.costFunction = ENCostFunctions.meanSquareError;
		this.step = true;
	}
	
	@Override
	public LearnResult learnOnLine(double realResult, double previousCost) {
		LearnResult out = new LearnResult();
//		out.realValue = realResult;
//		out.prediction = predict(false);
		out.errorValue = realResult - predict();
		out.costValue = previousCost + Math.pow(out.errorValue, 2);
		defaultRule(out.errorValue);
//		if(this.step) {
//			out.prediction = stepFunction(out.prediction);
//		}
		return out;
	}
	
	@Override
	public double predict() {
		double out = aditive();
		out = linear(out, false);
		return out;
	}

}
