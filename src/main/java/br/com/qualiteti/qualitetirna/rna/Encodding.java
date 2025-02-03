package br.com.qualiteti.qualitetirna.rna;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

public class Encodding {
	public static List<List<Double>> getOnehotEncoding (int dimensions){
		List<List<Double>> out = new ArrayList<>();
		List<Double> line;
		for(int i=0; i<dimensions; i++) {
			line = new ArrayList<>();
			for(int j=0; j<dimensions; j++) {
				line.add(j==i?1.0:0.0);
			}
			out.add(line);
		}
		return out;
	}
	
	public static List<Double> convertToOneHotEncoding(List<Double> values){
		List<Double> out = new ArrayList<>();
		double max = Double.MIN_VALUE;
		for(double v:values) {
			if(v>max){max = v;}
		}
		for(double v:values) {
			out.add(v==max?1.0:0.0);
		}
		
		return out;
	}
	
	public static Double[][] geraEmbeddings(int elementos, int d) {
	    Double[][] embeddings = new Double[elementos][d];
	    HashSet<String> uniqueCheck = new HashSet<>();
	    Random random = new Random(69);

	    for (int i = 0; i < elementos; i++) {
	        Double[] embedding;
	        String key;
	        do {
	            embedding = new Double[d];
	            StringBuilder sb = new StringBuilder();
	            for (int j = 0; j < d; j++) {
	                embedding[j] = -1 + (2 * random.nextDouble());
	                sb.append(embedding[j]).append(",");
	            }
	            key = sb.toString(); // Representação única do vetor
	        } while (uniqueCheck.contains(key));
	        uniqueCheck.add(key);
	        embeddings[i] = embedding;
	    }

	    return embeddings;
	}
}
