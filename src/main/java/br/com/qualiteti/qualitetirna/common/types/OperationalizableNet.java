package br.com.qualiteti.qualitetirna.common.types;

import java.util.ArrayList;
import java.util.List;

public class OperationalizableNet {
	public boolean hasInput = false;
	public boolean hasOutPut = false;
	public List<OperationalizableLayer> problematicLayers = new ArrayList<>();
	
	public boolean isOk() {
		return this.hasInput && this.hasOutPut && this.problematicLayers.isEmpty();
	}
}
