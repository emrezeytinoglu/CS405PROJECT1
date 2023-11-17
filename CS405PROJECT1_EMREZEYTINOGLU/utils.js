function multiplyMatrices(matrixA, matrixB) {
    var result = [];

    for (var i = 0; i < 4; i++) {
        result[i] = [];
        for (var j = 0; j < 4; j++) {
            var sum = 0;
            for (var k = 0; k < 4; k++) {
                sum += matrixA[i * 4 + k] * matrixB[k * 4 + j];
            }
            result[i][j] = sum;
        }
    }

    // Flatten the result array
    return result.reduce((a, b) => a.concat(b), []);
}
function createIdentityMatrix() {
    return new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]);
}
function createScaleMatrix(scale_x, scale_y, scale_z) {
    return new Float32Array([
        scale_x, 0, 0, 0,
        0, scale_y, 0, 0,
        0, 0, scale_z, 0,
        0, 0, 0, 1
    ]);
}

function createTranslationMatrix(x_amount, y_amount, z_amount) {
    return new Float32Array([
        1, 0, 0, x_amount,
        0, 1, 0, y_amount,
        0, 0, 1, z_amount,
        0, 0, 0, 1
    ]);
}

function createRotationMatrix_Z(radian) {
    return new Float32Array([
        Math.cos(radian), -Math.sin(radian), 0, 0,
        Math.sin(radian), Math.cos(radian), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_X(radian) {
    return new Float32Array([
        1, 0, 0, 0,
        0, Math.cos(radian), -Math.sin(radian), 0,
        0, Math.sin(radian), Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function createRotationMatrix_Y(radian) {
    return new Float32Array([
        Math.cos(radian), 0, Math.sin(radian), 0,
        0, 1, 0, 0,
        -Math.sin(radian), 0, Math.cos(radian), 0,
        0, 0, 0, 1
    ])
}

function getTransposeMatrix(matrix) {
    return new Float32Array([
        matrix[0], matrix[4], matrix[8], matrix[12],
        matrix[1], matrix[5], matrix[9], matrix[13],
        matrix[2], matrix[6], matrix[10], matrix[14],
        matrix[3], matrix[7], matrix[11], matrix[15]
    ]);
}

const vertexShaderSource = `
attribute vec3 position;
attribute vec3 normal; // Normal vector for lighting

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 normalMatrix;

uniform vec3 lightDirection;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vNormal = vec3(normalMatrix * vec4(normal, 0.0));
    vLightDirection = lightDirection;

    gl_Position = vec4(position, 1.0) * projectionMatrix * modelViewMatrix; 
}

`

const fragmentShaderSource = `
precision mediump float;

uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float shininess;

varying vec3 vNormal;
varying vec3 vLightDirection;

void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(vLightDirection);
    
    // Ambient component
    vec3 ambient = ambientColor;

    // Diffuse component
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor;

    // Specular component (view-dependent)
    vec3 viewDir = vec3(0.0, 0.0, 1.0); // Assuming the view direction is along the z-axis
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * specularColor;

    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}

`

/**
 * @WARNING DO NOT CHANGE ANYTHING ABOVE THIS LINE
 */



/**
 * 
 * @TASK1 Calculate the model view matrix by using the chatGPT
 */

function getChatGPTModelViewMatrix() {

    const transformationMatrix = new Float32Array([
        0.3535533845424652, -0.6123723983764648, 0.7071067690849304, 0.3,
        0.6123723983764648, 0.3535533845424652, 0.7071067690849304, -0.25,
        -0.7071067690849304, 0.7071067690849304, 4.371138828673793e-8, 0,
        0, 0, 0, 1
    ]);
    return getTransposeMatrix(transformationMatrix);
}


/**
 * 
 * @TASK2 Calculate the model view matrix by using the given 
 * transformation methods and required transformation parameters
 * stated in transformation-prompt.txt
 */
function getModelViewMatrix() {
    translated = createTranslationMatrix(0.3, -0.25, 0);
    scaled = createScaleMatrix(0.5, 0.5, 0.5);

    rotateX = createRotationMatrix_X(30);
    rotateY = createRotationMatrix_Y(45);
    rotateZ = createRotationMatrix_Z(60);

    matrix = multiplyMatrices(translated, scaled);
    matrix = multiplyMatrices(matrix, rotateX);
    matrix = multiplyMatrices(matrix, rotateY);
    matrix = multiplyMatrices(matrix, rotateZ);

    return matrix;
}

/**
 * 
 * @TASK3 Ask CHAT-GPT to animate the transformation calculated in 
 * task2 infinitely with a period of 10 seconds. 
 * First 5 seconds, the cube should transform from its initial 
 * position to the target position.
 * The next 5 seconds, the cube should return to its initial position.
 */


function getPeriodicMovement(startTime) {
    const animationDuration = 10 * 1000; // 10 seconds in milliseconds
    const elapsedTime = (Date.now() - startTime) % animationDuration;

    // Calculate the interpolation factor based on elapsed time
    const t = elapsedTime / animationDuration;

    // Interpolate translation along x, y, and z axes
    const translationMatrix = createTranslationMatrix(0.3 * t, -0.25 * t, 0.2 * Math.sin(Math.PI * t));

    // Interpolate scaling along all axes
    const scalingMatrix = createScaleMatrix(0.5 + 0.5 * t, 0.5 + 0.5 * t, 0.5 + 0.5 * t);

    // Adjust rotation speed for a smoother animation
    const rotationMatrixX = createRotationMatrix_X(5 * Math.sin(Math.PI * t));
    const rotationMatrixY = createRotationMatrix_Y(10 * Math.sin(Math.PI * t));
    const rotationMatrixZ = createRotationMatrix_Z(15 * Math.sin(Math.PI * t));

    // Combine the transformations
    const interpolatedMatrix = multiplyMatrices(translationMatrix, multiplyMatrices(rotationMatrixX, multiplyMatrices(rotationMatrixY, multiplyMatrices(rotationMatrixZ, scalingMatrix))));

    // Check if it's the first 5 seconds or the next 5 seconds
    if (elapsedTime < animationDuration / 2) {
        return interpolatedMatrix;
    } else {
        // During the next 5 seconds, interpolate from the current matrix back to the identity matrix
        const reverseTranslationMatrix = createTranslationMatrix(0.3 * (1 - t), -0.25 * (1 - t), -0.2 * Math.sin(Math.PI * (1 - t)));
        const reverseScalingMatrix = createScaleMatrix(0.5 + 0.5 * (1 - t), 0.5 + 0.5 * (1 - t), 0.5 + 0.5 * (1 - t));
        const reverseRotationMatrixX = createRotationMatrix_X(5 * Math.sin(Math.PI * (1 - t)));
        const reverseRotationMatrixY = createRotationMatrix_Y(10 * Math.sin(Math.PI * (1 - t)));
        const reverseRotationMatrixZ = createRotationMatrix_Z(15 * Math.sin(Math.PI * (1 - t)));

        const reverseInterpolationMatrix = multiplyMatrices(reverseTranslationMatrix, multiplyMatrices(reverseRotationMatrixX, multiplyMatrices(reverseRotationMatrixY, multiplyMatrices(reverseRotationMatrixZ, reverseScalingMatrix))));
        return reverseInterpolationMatrix;
    }
}

