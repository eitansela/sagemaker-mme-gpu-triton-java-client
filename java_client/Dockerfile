FROM openjdk:11

COPY credentials /root/.aws/credentials
COPY src/ src/
COPY gradle/ gradle/
COPY build.gradle settings.gradle gradlew ./

RUN ./gradlew assemble

ENTRYPOINT ["./gradlew","run"]