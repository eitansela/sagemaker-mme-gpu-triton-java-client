package org.example;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.example.pojo.IOTensor;
import org.example.pojo.InferenceResponse;
import org.example.pojo.Parameters;
import org.json.JSONArray;
import org.json.JSONObject;
import software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider;
import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient;
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest;
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointResponse;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;


public class InvokeTritonMultiModelEndpoint {

    static class Index {
        int start;
        int length;

        public Index(int start, int length) {
            this.start = start;
            this.length = length;
        }
    }

    private static final String HeaderLengthPrefix = "application/vnd.sagemaker-triton.binary+json;json-header-size=";

    private static final ObjectMapper jsonMapper = new ObjectMapper();

    public static void main(String[] args) throws JsonProcessingException {

        String endpointName = System.getenv("ENDPOINT_NAME");
        System.out.println("endpointName: "+endpointName);
        String targetModel = "e2e.tar.gz";
        System.out.println("targetModel: "+targetModel);

        JSONObject json = getJsonObject();
        String payload = json.toString();
        System.out.println("payload: "+payload);

        String contentType = "application/octet-stream";
        System.out.println("contentType: "+contentType);

        Region region = Region.US_EAST_1;
        SageMakerRuntimeClient runtimeClient = SageMakerRuntimeClient.builder()
                .region(region)
                .credentialsProvider(ProfileCredentialsProvider.create())
                .build();

        InvokeEndpointRequest endpointRequest = InvokeEndpointRequest.builder()
                .endpointName(endpointName)
                .contentType(contentType)
                .body(SdkBytes.fromString(payload, Charset.defaultCharset()))
                .targetModel(targetModel)
                .build();

        InvokeEndpointResponse response = runtimeClient.invokeEndpoint(endpointRequest);
        System.out.println("Model Response: "+response.body());
        System.out.println(response.body().asByteArray());

        byte[] bodyBytes = response.body().asByteArray();
        String bodyJson = new String(bodyBytes, Charset.defaultCharset());
        InferenceResponse responseObject = jsonMapper.readValue(bodyJson, InferenceResponse.class);

        // Construct name to binary index mapping.
        Map<String, Index> nameToBinaryIdx = new HashMap<>();
        int startPos = 0;
        for (IOTensor output : responseObject.getOutputs()) {
            Parameters param = output.getParameters();
            if (param == null) { continue; }
            Integer size = param.getInt(Parameters.KEY_BINARY_DATA_SIZE);
            if (size == null) { continue; }
            nameToBinaryIdx.put(output.getName(), new Index(startPos, size));
            startPos += size;
        }

        int headerLength = Integer. parseInt(response.contentType().substring(HeaderLengthPrefix.length()));
        int binaryDataSize = responseObject.getOutputByName("SENT_EMBED").getParameters().getInt("binary_data_size");

        float[] outputsData = getOutputAsFloat("SENT_EMBED", bodyBytes, headerLength, binaryDataSize, responseObject);

//        System.out.println("responseObject: "+responseObject.getOutputByName("SENT_EMBED"));
        System.out.println("outputsData: "+ Arrays.toString(outputsData));

    }

    private static JSONObject getJsonObject() {
        JSONObject json = new JSONObject();
        JSONArray inputs = new JSONArray();
        JSONObject inputItem = new JSONObject();
        inputItem.put("name","INPUT0");
        JSONArray shape = new JSONArray();
        shape.put(2);
        shape.put(1);
        inputItem.put("shape",shape);
        inputItem.put("datatype","BYTES");
        JSONArray data = new JSONArray();
        data.put("Sentence 1");
        data.put("Sentence 2");
        inputItem.put("data",data);
        inputs.put(inputItem);

        JSONArray outputs = new JSONArray();
        JSONObject outputItem = new JSONObject();
        outputItem.put("name","SENT_EMBED");
        JSONObject parameters = new JSONObject();
        JSONObject parametersTuple = new JSONObject();
        parametersTuple.put("binary_data",true);
        outputItem.put("parameters", parametersTuple);
        outputs.put(outputItem);

        json.put("inputs", inputs);
        json.put("outputs", outputs);
        return json;
    }


    public static float[] getOutputAsFloat(String output, byte[] binaryData, int headerLength, int binaryDataSize, InferenceResponse responseObject) {
        IOTensor out = responseObject.getOutputByName(output);
        if (out == null) {
            return null;
        }
        return (float[])getOutputImpl(out, binaryData, headerLength, binaryDataSize, float.class, ByteBuffer::getFloat);
    }

     private static <T> Object getOutputImpl(IOTensor out, byte[] binaryData, int headerLength, int binaryDataSize, Class<T> clazz, Function<ByteBuffer, T> getter) {

        long numElem = elemNumFromShape(out.getShape());
        Object array = Array.newInstance(clazz, (int)numElem);
        ByteBuffer buf = ByteBuffer.wrap(binaryData, headerLength, binaryDataSize);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < numElem; i++) {
            Array.set(array, i, getter.apply(buf));
        }
        return array;
    }

    public static long elemNumFromShape(long[] shape) {
        long ret = 1;
        for (long n : shape) {
            ret *= n;
        }
        return ret;
    }
}
