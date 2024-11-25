using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class MeshTests
    {
        [Test]
        public void TestDoubleAssignment()
        {
            var mesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            mesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            Assert.That(mesh.Points.Count == 8);
            Assert.That(mesh.Triangulation.Count == 12);
        }

        [Test]
        public void TestSaveLoad()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var tempFile = Path.GetTempFileName() + ".mrmesh";
            MeshSave.ToAnySupportedFormat(cubeMesh, tempFile);

            var readMesh = MeshLoad.FromAnySupportedFormat(tempFile);
            Assert.That(cubeMesh.Points.Count == readMesh.Points.Count);
            Assert.That(cubeMesh.Triangulation.Count == readMesh.Triangulation.Count);

            File.Delete(tempFile);
        }

        [Test]
        public void TestSaveLoadException()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var tempFile = Path.GetTempFileName() + ".fakeextension";

            try
            {
                MeshSave.ToAnySupportedFormat(cubeMesh, tempFile);
            }
            catch (System.Exception e)
            {
                Assert.That(e.Message.Contains("unsupported file extension"));
            }

            try
            {
                MeshLoad.FromAnySupportedFormat(tempFile);
            }
            catch (System.Exception e)
            {
                Assert.That(e.Message.Contains("unsupported file extension"));
            }
        }

        [Test]
        public void TestSaveLoadCtm()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var tempFile = Path.GetTempFileName() + ".ctm";
            MeshSave.ToAnySupportedFormat(cubeMesh, tempFile);

            var readMesh = MeshLoad.FromAnySupportedFormat(tempFile);
            Assert.That(readMesh.Points.Count == 8);
            Assert.That(readMesh.Triangulation.Count == 12);

            File.Delete(tempFile);
        }

        [Test]
        public void TestFromTriangles()
        {
            List<Vector3f> points = new List<Vector3f>();
            points.Add(new Vector3f(0, 0, 0));
            points.Add(new Vector3f(0, 1, 0));
            points.Add(new Vector3f(1, 1, 0));
            points.Add(new Vector3f(1, 0, 0));
            points.Add(new Vector3f(0, 0, 1));
            points.Add(new Vector3f(0, 1, 1));
            points.Add(new Vector3f(1, 1, 1));
            points.Add(new Vector3f(1, 0, 1));

            List<ThreeVertIds> triangles = new List<ThreeVertIds>();
            triangles.Add(new ThreeVertIds(0, 1, 2));
            triangles.Add(new ThreeVertIds(2, 3, 0));
            triangles.Add(new ThreeVertIds(0, 4, 5));
            triangles.Add(new ThreeVertIds(5, 1, 0));
            triangles.Add(new ThreeVertIds(0, 3, 7));
            triangles.Add(new ThreeVertIds(7, 4, 0));
            triangles.Add(new ThreeVertIds(6, 5, 4));
            triangles.Add(new ThreeVertIds(4, 7, 6));
            triangles.Add(new ThreeVertIds(1, 5, 6));
            triangles.Add(new ThreeVertIds(6, 2, 1));
            triangles.Add(new ThreeVertIds(6, 7, 3));
            triangles.Add(new ThreeVertIds(3, 2, 6));

            var mesh = Mesh.FromTriangles(points, triangles);
            Assert.That(mesh.Points.Count == 8);
            Assert.That(mesh.Triangulation.Count == 12);
        }

        [Test]
        public void TestEmptyFile()
        {
            string path = Path.GetTempFileName() + ".mrmesh";
            var file = File.Create(path);
            file.Close();
            Assert.Throws<SystemException>(() => MeshLoad.FromAnySupportedFormat(path));
            File.Delete(path);
        }

        [Test]
        public void TestTransform()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var xf = new AffineXf3f(Vector3f.Diagonal(1.0f));
            cubeMesh.Transform(xf);

            Assert.That(cubeMesh.Points[0] == new Vector3f(0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.Points[1] == new Vector3f(0.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.Points[2] == new Vector3f(1.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.Points[3] == new Vector3f(1.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.Points[4] == new Vector3f(0.5f, 0.5f, 1.5f));
            Assert.That(cubeMesh.Points[5] == new Vector3f(0.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.Points[6] == new Vector3f(1.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.Points[7] == new Vector3f(1.5f, 0.5f, 1.5f));
        }

        [Test]
        public void TestTransformWithRegion()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var region = new BitSet(8);
            region.Set(0);
            region.Set(2);
            region.Set(4);
            region.Set(6);

            var xf = new AffineXf3f(Vector3f.Diagonal(1.0f));
            cubeMesh.Transform(xf, region);

            Assert.That(cubeMesh.Points[0] == new Vector3f(0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.Points[1] == new Vector3f(-0.5f, 0.5f, -0.5f));
            Assert.That(cubeMesh.Points[2] == new Vector3f(1.5f, 1.5f, 0.5f));
            Assert.That(cubeMesh.Points[3] == new Vector3f(0.5f, -0.5f, -0.5f));
            Assert.That(cubeMesh.Points[4] == new Vector3f(0.5f, 0.5f, 1.5f));
            Assert.That(cubeMesh.Points[5] == new Vector3f(-0.5f, 0.5f, 0.5f));
            Assert.That(cubeMesh.Points[6] == new Vector3f(1.5f, 1.5f, 1.5f));
            Assert.That(cubeMesh.Points[7] == new Vector3f(0.5f, -0.5f, 0.5f));
        }

        [Test]
        public void TestLeftTriVerts()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var triVerts = cubeMesh.GetLeftTriVerts(new EdgeId(0));
            Assert.That(triVerts[0].Id, Is.EqualTo(0));
            Assert.That(triVerts[1].Id, Is.EqualTo(1));
            Assert.That(triVerts[2].Id, Is.EqualTo(2));

            triVerts = cubeMesh.GetLeftTriVerts(new EdgeId(6));
            Assert.That(triVerts[0].Id, Is.EqualTo(2));
            Assert.That(triVerts[1].Id, Is.EqualTo(3));
            Assert.That(triVerts[2].Id, Is.EqualTo(0));
        }

        [Test]
        public void TestSaveLoadToObj()
        {
            Assert.DoesNotThrow(() =>
            {
                var objects = new List<NamedMeshXf>();
                var obj = new NamedMeshXf();
                obj.mesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
                obj.name = "Cube";
                obj.toWorld = new AffineXf3f(Vector3f.Diagonal(1));
                objects.Add(obj);

                obj.mesh = Mesh.MakeSphere(1.0f, 100);
                obj.name = "Sphere";
                obj.toWorld = new AffineXf3f(Vector3f.Diagonal(-2));
                objects.Add(obj);

                var tempFile = Path.GetTempFileName() + ".obj";
                MeshSave.SceneToObj(objects, tempFile);

                var settings = new ObjLoadSettings();
                var loadedObjs = MeshLoad.FromSceneObjFile(tempFile, false, settings);
                Assert.That(loadedObjs.Count == 2);

                var loadedMesh = loadedObjs[0].mesh;
                var loadedXf = loadedObjs[0].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if ( loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.Points.Count == 8);
                Assert.That(loadedObjs[0].name == "Cube");
                Assert.That(loadedXf.B.X == 0.0f);

                loadedMesh = loadedObjs[1].mesh;
                loadedXf = loadedObjs[1].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if (loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.Points.Count == 100);
                Assert.That(loadedObjs[1].name == "Sphere");
                Assert.That(loadedXf.B.X == 0.0f);

                settings.customXf = true;
                loadedObjs = MeshLoad.FromSceneObjFile(tempFile, false, settings);
                Assert.That(loadedObjs.Count == 2);

                loadedMesh = loadedObjs[0].mesh;
                loadedXf = loadedObjs[0].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if (loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.Points.Count == 8);
                Assert.That(loadedObjs[0].name == "Cube");
                Assert.That(loadedXf.B.X == 1.0f);

                loadedMesh = loadedObjs[1].mesh;
                loadedXf = loadedObjs[1].xf;
                Assert.That(loadedMesh is not null);
                Assert.That(loadedXf is not null);
                if (loadedMesh is null || loadedXf is null)
                    return;

                Assert.That(loadedMesh.Points.Count == 100);
                Assert.That(loadedObjs[1].name == "Sphere");
                Assert.That(loadedXf.B.X == -2.0f);

                loadedMesh = loadedObjs[0].mesh;
                if (loadedMesh is not null) 
                    loadedMesh.Dispose();

                loadedMesh = loadedObjs[1].mesh;
                if (loadedMesh is not null)
                    loadedMesh.Dispose();

                File.Delete(tempFile);
            });
        }

        [Test]
        public void TestCalculatingVolume()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            Assert.That(cubeMesh.Volume(), Is.EqualTo(1.0).Within(1e-6));

            var validPoints = new FaceBitSet(8);
            validPoints.Set(0);
            validPoints.Set(1);
            validPoints.Set(3);
            validPoints.Set(4);
            validPoints.Set(5);
            validPoints.Set(7);

            Assert.That(cubeMesh.Volume(validPoints), Is.EqualTo(0.5).Within(1e-6));
        }

        [Test]
        public void TestProjection()
        {
            var p = new Vector3f(1, 2, 3);
            var mp = new MeshPart(Mesh.MakeSphere(1.0f, 1000));
            var projRes = Mesh.FindProjection(p, mp);
            Assert.That(projRes.distanceSquared, Is.EqualTo(7.529f).Within(1e-3));

            Assert.That(projRes.pointOnFace.faceId.Id, Is.EqualTo(904));
            Assert.That(projRes.pointOnFace.point.X, Is.EqualTo(0.310).Within(1e-3));
            Assert.That(projRes.pointOnFace.point.Y, Is.EqualTo(0.507).Within(1e-3));
            Assert.That(projRes.pointOnFace.point.Z, Is.EqualTo(0.803).Within(1e-3));

            Assert.That(projRes.meshTriPoint.e.Id, Is.EqualTo(1640));
            Assert.That(projRes.meshTriPoint.bary.a, Is.EqualTo(0.053).Within(1e-3));
            Assert.That(projRes.meshTriPoint.bary.b, Is.EqualTo(0.946).Within(1e-3));

            var xf = new AffineXf3f(Vector3f.Diagonal(1.0f));
            projRes = Mesh.FindProjection(p, mp, float.MaxValue, xf);

            Assert.That(projRes.pointOnFace.faceId.Id, Is.EqualTo(632));
            Assert.That(projRes.pointOnFace.point.X, Is.EqualTo(1.000).Within(1e-3));
            Assert.That(projRes.pointOnFace.point.Y, Is.EqualTo(1.439).Within(1e-3));
            Assert.That(projRes.pointOnFace.point.Z, Is.EqualTo(1.895).Within(1e-3));

            Assert.That(projRes.meshTriPoint.e.Id, Is.EqualTo(1898));
            Assert.That(projRes.meshTriPoint.bary.a, Is.EqualTo(0.5).Within(1e-3));
            Assert.That(projRes.meshTriPoint.bary.b, Is.EqualTo(0.0).Within(1e-3));
        }

        [Test]
        public void TestValidPoints()
        {
            Assert.DoesNotThrow(() =>
            {
                var mesh = Mesh.MakeSphere(1.0f, 3000);
                var count = mesh.ValidPoints.Count();
                Assert.That(count, Is.EqualTo(3000));
                mesh.Dispose();
            });
        }

        [Test]
        public void TestClone()
        {
            var mesh = Mesh.MakeSphere(1.0f, 3000);
            var clone = mesh.Clone();
            Assert.That(clone, Is.Not.SameAs(mesh));
            Assert.That(clone.Points.Count, Is.EqualTo(mesh.Points.Count));
            Assert.That(clone.Triangulation.Count, Is.EqualTo(mesh.Triangulation.Count));
            mesh.Dispose();
            clone.Dispose();
        }

        [Test]
        public void TestUniteCloseVertices()
        {
            var mesh = Mesh.MakeSphere(1.0f, 3000);
            Assert.That(mesh.ValidPoints.Count() == 3000);
            var old2new = new List<VertId>();
            var unitedCount = MeshBuilder.UniteCloseVertices(ref mesh, 0.1f, false, old2new);
            Assert.That(unitedCount, Is.EqualTo(2230));
            Assert.That(old2new[1000].Id, Is.EqualTo(42));
            Assert.That(mesh.ValidPoints.Count() < 3000);
            mesh.Dispose();
        }

        [Test]
        public void TestArea()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            Assert.That(cubeMesh.Area(), Is.EqualTo(6.0).Within(0.001));

            var faces = new FaceBitSet(12, true);
            for (int i = 0; i < 6; ++i)
                faces.Set(i, false);

            Assert.That(cubeMesh.Area(faces), Is.EqualTo(3.0).Within(0.001));

            cubeMesh.DeleteFaces(faces);
            Assert.That(cubeMesh.Area(), Is.EqualTo(3.0).Within(0.001));
        }

        [Test]
        public void TestIncidentFacesFromVerts()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var verts = new VertBitSet(8, false);
            verts.Set(0, true);
            var faces = RegionBoundary.GetIncidentFaces(cubeMesh, verts);
            Assert.That(faces.Count, Is.EqualTo(6));
        }

        [Test]
        public void TestIncidentFacesFromEdges()
        {
            var cubeMesh = Mesh.MakeCube(Vector3f.Diagonal(1), Vector3f.Diagonal(-0.5f));
            var edges = new UndirectedEdgeBitSet(12, false);
            edges.Set(0, true);
            var faces = RegionBoundary.GetIncidentFaces(cubeMesh, edges);
            Assert.That(faces.Count, Is.EqualTo(8));
        }

        [Test]
        public void TestShortEdges()
        {
            var mesh = Mesh.MakeTorus(1.0f, 0.05f, 16, 16);
            var shortEdges = MeshFixer.FindShortEdges(new MeshPart( mesh ), 0.1f);
            Assert.That(shortEdges.Count(), Is.EqualTo(256));
        }
    }    
}

