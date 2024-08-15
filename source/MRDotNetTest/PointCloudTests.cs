﻿using System;
using System.IO;
using System.Collections.Generic;
using NUnit.Framework;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class PoitCloudTests
    {
        static PointCloud MakeCube()
        {
            var points = new PointCloud();
            points.AddPoint(new Vector3f(0, 0, 0));
            points.AddPoint(new Vector3f(0, 1, 0));
            points.AddPoint(new Vector3f(1, 1, 0));
            points.AddPoint(new Vector3f(1, 0, 0));
            points.AddPoint(new Vector3f(0, 0, 1));
            points.AddPoint(new Vector3f(0, 1, 1));
            points.AddPoint(new Vector3f(1, 1, 1));
            points.AddPoint(new Vector3f(1, 0, 1));
            return points;
        }

        [Test]
        public void TestPointCloud()
        {
            var points = MakeCube();

            Assert.That(points.Points.Count == 8);
            Assert.That(points.Normals.Count == 0);

            var bbox = points.BoundingBox;
            Assert.That(bbox.Min == new Vector3f(0, 0, 0));
            Assert.That(bbox.Max == new Vector3f(1, 1, 1));
        }

        [Test]
        public void TestPointCloudNormals()
        {
            var points = new PointCloud();
            points.AddPoint(new Vector3f(0, 0, 0), new Vector3f(0, 0, 1));
            points.AddPoint(new Vector3f(0, 1, 0), new Vector3f(0, 0, 1));

            Assert.That(points.Points.Count == 2);
            Assert.That(points.Points.Count == 2);
        }

        [Test]
        public void TestNormalsError()
        {
            var points = new PointCloud();
            points.AddPoint(new Vector3f(0, 0, 0), new Vector3f(0, 0, 1));
            Assert.Throws<InvalidOperationException>(() => points.AddPoint(new Vector3f(0, 0, 0)));

            points = new PointCloud();
            points.AddPoint(new Vector3f(0, 0, 0));
            Assert.Throws<InvalidOperationException>(() => points.AddPoint(new Vector3f(0, 0, 0), new Vector3f(0, 0, 1)));
        }

        [Test]
        public void TestSaveLoad()
        {
            var points = MakeCube();
            var tempFile = Path.GetTempFileName() + ".ply";
            PointCloud.ToAnySupportedFormat(points, tempFile);

            var readPoints = PointCloud.FromAnySupportedFormat(tempFile);
            Assert.That(points.Points.Count == readPoints.Points.Count);
        }

        [Test]
        public void TestEmptyFile()
        {
            string path = Path.GetTempFileName() + ".ply";
            var file = File.Create(path);
            file.Close();
            Assert.Throws<SystemException>(() => PointCloud.FromAnySupportedFormat(path));
            File.Delete(path);
        }

        [Test]
        public void TestNullArgs()
        {
            Assert.Throws<ArgumentNullException>(() => PointCloud.FromAnySupportedFormat(null));
            Assert.Throws<ArgumentNullException>(() => PointCloud.ToAnySupportedFormat(null, null));
            var points = new PointCloud();
            Assert.Throws<ArgumentNullException>(() => points.AddPoint(null));
            Assert.Throws<ArgumentNullException>(() => points.AddPoint(null, null));
        }
    }
}