package br.com.qualiteti.qualitetirna.rna.definition;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.enums.EnLayerType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;

public class LayerDefinition implements Serializable {
	private static final long serialVersionUID = 1L;
	public EnLayerType layerType;
	public EnNeuronType neuronsType;
	public int netOrder;
	public double layerLearningRate = 1e2;
	public List<NeuronDefinition> neuronsDefinitions = new ArrayList<>();
}
