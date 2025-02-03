package br.com.qualiteti.qualitetirna.rna.serialization;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import br.com.qualiteti.qualitetirna.rna.definition.NetDefinition;

public class NeuralNetSerialization {
	// Serializa um objeto NetDefinition para um vetor de bytes
	public static byte[] serialize(NetDefinition neuralNetDefinition) throws IOException {
		try (ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
			 ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream)) {
			objectOutputStream.writeObject(neuralNetDefinition);
			return byteArrayOutputStream.toByteArray();
		}
	}

	// Desserializa um vetor de bytes para um objeto NetDefinition
	public static NetDefinition deserialize(byte[] bytes) throws IOException, ClassNotFoundException {
		try (ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes);
			 ObjectInputStream objectInputStream = new ObjectInputStream(byteArrayInputStream)) {
			return (NetDefinition) objectInputStream.readObject();
		}
	}
}
