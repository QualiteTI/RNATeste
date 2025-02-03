package br.com.qualiteti.qualitetirna.common.types;

import java.util.ArrayList;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.enums.EnLayerType;

public class OperationalizableLayer {
	public int netOrder;
	public EnLayerType type;
	public boolean hasNeurons = false;
	public List<OperationalizableNeuron> prolematicsNeurons = new ArrayList<>();
	
	public boolean isOk() {
		return this.hasNeurons && (this.prolematicsNeurons.isEmpty());
	}
}
