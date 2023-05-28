//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : SAGI BENEDEK
// Neptun : ECSGGY
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
float maxH = -INFINITY;
float minH = +INFINITY;
template<class T> struct Dnum {

	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 100;

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)(windowWidth / 2) / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	}
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};


struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
};

class CheckerBoardTexture : public Texture {
public:
	CheckerBoardTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};


struct RenderState {
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};


class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.ks, name + ".ks");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};


class GouraudShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;
 
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};
 
		uniform mat4  MVP, M, Minv;  
		uniform Light[8] lights;      
		uniform int   nLights;		 
		uniform vec3  wEye;         
		uniform float minimum;
		uniform float maximum;
		uniform Material  material;  
		layout(location = 0) in vec3  vtxPos;          
		layout(location = 1) in vec3  vtxNorm;      	
 
		out vec3 radiance;		    
 
		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; 
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;	
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
 
			float height = vtxPos.z; // Assuming the height is stored in the y-component of vtxPos
 
			radiance = vec3(0, 0, 0);
			float t = (height - minimum) / (maximum - minimum);
 
			t = clamp(t, 0.0, 1.0); 
 
			vec3 greenColor = vec3(0, 0.5, 0);
			vec3 brownColor = vec3(0.4, 0.2, 0);
			vec3 blackColor = vec3(0, 0, 0);
 
			if (t < 0.75) {
				// Green to brown interpolation
				radiance = mix(greenColor, brownColor, t / 0.75);
			} else {
				// Brown to black interpolation
				radiance = mix(brownColor, blackColor, (t - 0.75) / 0.25);
			}
 
 
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += lights[i].Le * cost + lights[i].Le * material.ks * pow(cosd, material.shininess);
			}
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;
 
		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer
 
		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(minH, "minimum");
		setUniform(maxH, "maximum");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

class Geometry {
protected:
	unsigned int vao, vbo;
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

class RectangularCuboid : public ParamSurface {
private:
	float width, height, depth;  // Dimensions of the cuboid

public:
	RectangularCuboid(float width, float height, float depth)
		: width(width), height(height), depth(depth) {
		create();
	}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) override {
		U = U * width * 2.0f;
		V = V * height;
		X = U - width / 2.0f;
		Y = V - height / 2.0f;
		Z = depth / 2.0f;
	}
};

class Terrain : public ParamSurface {
	int n = 12;
	float phiMatrix[12][12];
	float AMatrix[12][12];
public:
	Terrain() {
		for (int f1 = 0; f1 < n; f1++)
		{
			for (int f2 = 0; f2 < n; f2++)
			{
				float phi = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)) * M_PI * 2;
				float af1f2;
				if (f1 + f2 > 0)
					af1f2 = 0.2 / sqrtf(f1 * f1 + f2 * f2);
				else
					af1f2 = 0.0;
				phiMatrix[f1][f2] = phi;
				AMatrix[f1][f2] = af1f2;
			};
		}
		create();
	}
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z)
	{
		U = U * 2.0f * M_PI - M_PI,
			V = V * 2.0f * M_PI - M_PI;
		X = U;
		Y = V;
		float height = 0.0;
		for (int f1 = 0; f1 < n; f1++)
			for (int f2 = 0; f2 < n; f2++)
				height = height + AMatrix[f1][f2] * cosf(f1 * X.f + f2 * Y.f + phiMatrix[f1][f2]);
		Z = height;
		if (Z.f < minH) minH = Z.f;
		if (Z.f > maxH) maxH = Z.f;

	}
};



struct Object {
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend; }
};

//---------------------------
class Scene {
	//---------------------------
	std::vector<Object*> objects;
	std::vector<Camera> cameras;
	std::vector<Light> lights;
public:
	void Build() {
		Shader* gouraudShader = new GouraudShader();

		Material* material0 = new Material;
		material0->kd = vec3(.5, .5, .5);
		material0->ks = vec3(0.5, 0.5, 0.5);
		material0->ka = vec3(.5, .5, .5);
		material0->shininess = 30;

		Material* material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30;

		Texture* texture15x20 = new CheckerBoardTexture(15, 20);

		Geometry* terrain = new Terrain();
		Geometry* rectangular = new RectangularCuboid(1, 1, 1);
		Geometry* square = new RectangularCuboid(0.5, 1, 1);

		// Create objects by setting up their vertex data on the GPU
		Object* terrainObject1 = new Object(gouraudShader, material0, texture15x20, terrain);
		terrainObject1->rotationAxis = vec3(1, 0, 0);
		terrainObject1->rotationAngle = -M_PI / 2;
		objects.push_back(terrainObject1);
		Object* rectangularObject1 = new Object(gouraudShader, material0, texture15x20, rectangular);
		rectangularObject1->translation = vec3(0, 5, 0);


		Object* rectangularObject2 = new Object(gouraudShader, material0, texture15x20, rectangular);
		rectangularObject2->translation = vec3(0, 5, 0);
		rectangularObject2->rotationAngle = M_PI / 2;
		rectangularObject2->rotationAxis = vec3(1, 0, 0);

		Object* rectangularObject3 = new Object(gouraudShader, material0, texture15x20, rectangular);
		rectangularObject3->translation = vec3(0, 0, -1);

		Object* rectangularObject4 = new Object(gouraudShader, material0, texture15x20, rectangular);
		rectangularObject4->translation = vec3(0, 6, 0);
		rectangularObject4->rotationAngle = M_PI / 2;
		rectangularObject4->rotationAxis = vec3(1, 0, 0);

		Object* squarerObject1 = new Object(gouraudShader, material0, texture15x20, square);
		squarerObject1->translation = vec3(1, 5, .25);
		squarerObject1->rotationAngle = M_PI / 2;
		squarerObject1->rotationAxis = vec3(0, 1, 0);
		Object* squarerObject2 = new Object(gouraudShader, material0, texture15x20, square);
		squarerObject2->translation = vec3(-1, 5, .25);
		squarerObject2->rotationAngle = M_PI / 2;
		squarerObject2->rotationAxis = vec3(0, 1, 0);


		objects.push_back(rectangularObject1);
		objects.push_back(rectangularObject2);
		objects.push_back(rectangularObject3);
		objects.push_back(rectangularObject4);
		objects.push_back(squarerObject1);
		objects.push_back(squarerObject2);




		Camera camera1;
		Camera camera2;
		// Camera1
		camera1.wEye = vec3(10, 3, 0);
		camera1.wLookat = vec3(0, 0, 0);
		camera1.wVup = vec3(0, 1, 0);
		cameras.push_back(camera1);
		// Camera2
		camera2.wEye = vec3(-10, 10, 4);
		camera2.wLookat = vec3(0, 0, 0);
		camera2.wVup = vec3(0, 1, 0);
		cameras.push_back(camera2);

		lights.resize(1);
		lights[0].wLightPos = vec4(-5, 5, 4, 0);
		lights[0].La = vec3(0.1f, 0.1f, 1);
		lights[0].Le = vec3(0.1, 0.1, 0.1);
	}

	void Render(int index) {
		RenderState state;
		state.wEye = cameras[index].wEye;
		state.V = cameras[index].V();
		state.P = cameras[index].P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		//for (Object* obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {

	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	glViewport(0, 0, windowWidth / 2, windowHeight);
	scene.Render(0);
	glViewport(300, 0, windowWidth / 2, windowHeight);
	scene.Render(1);
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}