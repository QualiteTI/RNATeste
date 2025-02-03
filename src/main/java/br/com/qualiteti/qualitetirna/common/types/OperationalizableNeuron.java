package br.com.qualiteti.qualitetirna.common.types;

import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronSuper;
import lombok.NoArgsConstructor;

@NoArgsConstructor
public class OperationalizableNeuron {
	public int neuronOrder;
	public EnNeuronType neuronType;
	public EnActivationType activationType;
	public boolean hasWeights = false;
	public boolean hasBias = false;
	public boolean hasEntries = false;
	public boolean qntEntriesAndBias = false;
	public boolean hasAlfaReLU = false;
	
	public OperationalizableNeuron(int order, NeuronSuper n) {
		this.neuronOrder = order;
		this.neuronType = n.type;
		this.activationType = n.activationFunction;
		this.hasWeights = !n.weights.isEmpty();
		this.hasBias = (n.bias != 0);
		this.hasEntries = !n.entries.isEmpty();
		this.qntEntriesAndBias = (n.weights.size() == n.entries.size());
		this.hasAlfaReLU = (n.alfaReLU != 0);
	}
	
	public ExtendedBoolean isOk(boolean inInputLayer) {
		ExtendedBoolean out = new ExtendedBoolean(true, "");
		String msg = "";
		if(inInputLayer) {
			if(!this.hasEntries) {msg += "O neurônio não tem valores de entrada. ";}
			if(!this.qntEntriesAndBias) {msg += "O neurônio não tem a mesma quantidade de pesos e valores. ";}
		}
		if(!this.hasWeights) {msg += "O neurônio não tem pesos. ";}
		if(!this.hasBias) {msg += "O neurônio não tem Bias(Viés). ";}
		if((this.activationType == EnActivationType.LeakyReLU || this.activationType == EnActivationType.eLU) && !this.hasAlfaReLU) {
			msg += "O neurônio não tem alfaReLU e isso é necessário para a função de ativação definida. ";
		}
		if(!msg.equals("")) {
			out.setValues(false, msg.trim());
		}
		return out;
	}
	
}
