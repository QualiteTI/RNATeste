package br.com.qualiteti.qualitetirna;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import br.com.qualiteti.qualitetirna.common.enums.ENCostFunctions;
import br.com.qualiteti.qualitetirna.common.enums.EnActivationType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronDerivativeType;
import br.com.qualiteti.qualitetirna.common.enums.EnNeuronType;
import br.com.qualiteti.qualitetirna.common.types.LearnResult;
import br.com.qualiteti.qualitetirna.common.ClasseUtils;
import br.com.qualiteti.qualitetirna.rna.Acuracy;
import br.com.qualiteti.qualitetirna.rna.CostOffLine;
import br.com.qualiteti.qualitetirna.rna.DataNormalizer;
import br.com.qualiteti.qualitetirna.rna.Layer;
import br.com.qualiteti.qualitetirna.rna.NeuralNet;
import br.com.qualiteti.qualitetirna.rna.definition.NetDefinition;
import br.com.qualiteti.qualitetirna.rna.neurons.Neuron;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronAdaline;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronPerceptron;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronSigmoid;
import br.com.qualiteti.qualitetirna.rna.neurons.NeuronSuper;
import br.com.qualiteti.qualitetirna.rna.serialization.NeuralNetSerialization;
import br.com.qualiteti.qualitetirna.testes.TestesImplement;

@SpringBootApplication
public class QualiteRnaApplication implements CommandLineRunner  {

	public static void main(String[] args) {
		SpringApplication.run(QualiteRnaApplication.class, args);
	}

	@Override
    public void run(String... args) throws Exception {
		TestesImplement.testeSigmoidFase1();
	}
	
	


}
