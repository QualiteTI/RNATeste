package br.com.qualiteti.qualitetirna.rna.definition;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;

public class NetDefinition implements Serializable {

	private static final long serialVersionUID = 1L;
	public ENCostFunctions costFunction;
	public List<LayerDefinition> layersDefinitions = new ArrayList<>();
}
