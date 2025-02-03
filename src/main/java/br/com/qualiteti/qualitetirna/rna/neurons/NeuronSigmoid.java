package br.com.qualiteti.qualitetirna.rna.neurons;

import java.util.ArrayList;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;
import br.com.qualiteti.qualitetirna.common.types.LearnResult;
import br.com.qualiteti.qualitetirna.rna.CostOffLine;

public class NeuronSigmoid extends NeuronSuper {

	public NeuronSigmoid() {
		super(EnNeuronType.sigmoid);
		this.activationFunction = EnActivationType.Sigmoid;
		this.costFunction = ENCostFunctions.crossEntropy;
		this.step = false;
	}
	
	@Override
	public LearnResult learnOffLine(List<Double> realResults, List<Double> predicts, List<List<Double>> data) {
		LearnResult out = new LearnResult();
		double sum = 0;
		double totalError = 0;
		//List<Double> error = new ArrayList<>();
		List<Double> gradient = new ArrayList<>();
		
		if(realResults.size() != predicts.size()) {
			return null;
		}
		 
        for (int j = 0; j < this.weights.size(); j++) { // Iterar sobre colunas de x (tamanho de w)
            sum = 0.0;
            for (int i = 0; i < realResults.size(); i++) { // Iterar sobre linhas de error/x
                sum += (realResults.get(i)-predicts.get(i)) * data.get(i).get(j);
                totalError += (realResults.get(i)-predicts.get(i));
            }
            gradient.add(sum);
        }

        // Multiplica pelo learning_rate e soma em w
        for (int j = 0; j < this.weights.size(); j++) {
        	this.weights.set(j, this.weights.get(j) + this.learningRateW * gradient.get(j));
        }
        this.bias = this.bias + this.learningRateB * totalError;
		
		out.costValue = CostOffLine.crossEntropy(predicts, realResults);
		out.errorValue = totalError;
		
		return out;
	}
	
	@Override
	public double predict() {
		double out = aditive();
		out = sigmoid(out, false);
		return this.step?this.stepFunction(out):out;
	}
	
}
