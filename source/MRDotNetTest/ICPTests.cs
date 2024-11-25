using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class ICPTests
    {
        [Test]
        public void TestMultiwayICP()
        {
            List<MeshOrPointsXf> inputs = new List<MeshOrPointsXf>(2);
            Box3f maxBBox = new Box3f();

            inputs.Add( new MeshOrPointsXf(Mesh.MakeSphere(1.0f, 1000), new AffineXf3f()));
            Box3f bbox = inputs[0].obj.BoundingBox;
            if (!maxBBox.Valid() || bbox.Volume() > maxBBox.Volume())
                maxBBox = bbox;

            inputs.Add(new MeshOrPointsXf(Mesh.MakeSphere(1.0f, 1000), new AffineXf3f(Matrix3f.Rotation(Vector3f.PlusZ(), 0.1f))));
            bbox = inputs[1].obj.BoundingBox;
            if (!maxBBox.Valid() || bbox.Volume() > maxBBox.Volume())
                maxBBox = bbox;

            MultiwayICPSamplingParameters samplingParams = new MultiwayICPSamplingParameters();
            samplingParams.samplingVoxelSize = maxBBox.Diagonal() * 0.03f;

            MultiwayICP icp = new MultiwayICP(inputs, samplingParams);
            ICPProperties iCPProperties = new ICPProperties();
            icp.SetParams(iCPProperties);
            icp.UpdateAllPointPairs();

            Assert.That(icp.GetNumActivePairs(), Is.EqualTo(1748));
            Assert.That(icp.GetNumSamples(), Is.EqualTo(1748));
            Assert.That(icp.GetMeanSqDistToPoint(), Is.EqualTo(0.00254).Within(1e-5));

            Console.WriteLine("Calculating transformations...");
            var xfs = icp.CalculateTransformations();
            Assert.That(icp.GetNumActivePairs(), Is.EqualTo(1748));
            Assert.That(icp.GetNumSamples(), Is.EqualTo(1748));
            Assert.That(icp.GetMeanSqDistToPoint(), Is.EqualTo(0.00226).Within(1e-5));
        }

        [Test]
        public void TestICP()
        {
            var torusRef = Mesh.MakeTorus(2, 1, 32, 32);
            var torusMove = Mesh.MakeTorus(2, 1, 32, 32);

            var axis = Vector3f.PlusX();
            var trans = new Vector3f(0.0f, 0.2f, 0.105f);
            var xf = new AffineXf3f(Matrix3f.Rotation(axis, 0.2f), trans);

            MeshOrPointsXf flt = new MeshOrPointsXf(torusMove, xf );
            MeshOrPointsXf refer = new MeshOrPointsXf( torusRef, new AffineXf3f());

            var fltSamples = torusMove.ValidPoints as VertBitSet;
            var referSamples = torusRef.ValidPoints as VertBitSet;
            Assert.That(fltSamples is not null);
            Assert.That(referSamples is not null);

            if (fltSamples is null || referSamples is null)
                return;

            var icp = new ICP(flt, refer, fltSamples, referSamples);

            var newXf = icp.CalculateTransformation();
            Console.WriteLine(icp.GetStatusInfo());

            var diffXf = new AffineXf3f();
            diffXf.A -= newXf.A;
            diffXf.B -= newXf.B;

            Assert.That(Math.Abs(diffXf.A.X.X), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.X.Y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.X.Z), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Y.X), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Y.Y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Y.Z), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Z.X), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Z.Y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.A.Z.Z), Is.LessThan(1e-6f));

            Assert.That(Math.Abs(diffXf.B.X), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.B.Y), Is.LessThan(1e-6f));
            Assert.That(Math.Abs(diffXf.B.Z), Is.LessThan(1e-6f));

            var pairs = icp.GetRef2FltPairs();
            Assert.That(pairs.pairs.Count, Is.EqualTo(1024));

            pairs = icp.GetFlt2RefPairs();
            Assert.That(pairs.pairs.Count, Is.EqualTo(1024));
        }
    }
}
