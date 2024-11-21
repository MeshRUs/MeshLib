﻿using System;
using System.IO;
using NUnit.Framework;

using MR.DotNet;

namespace MR.DotNet.Test
{
    [TestFixture]
    internal class DecimateTests
    {
        [Test]
        public void TestDecimate()
        {
            var sphere = Mesh.MakeSphere( 0.5f, 30000 );

            BitSetReadOnly savedRegion = sphere.ValidFaces.Clone();

            var parameters = new DecimateParameters();
            FaceBitSet? region = sphere.ValidFaces as FaceBitSet;
            if ( region is not null )
                parameters.region = region.Clone() as FaceBitSet;

            parameters.maxTriangleAspectRatio = 80;

            var decimateResult = MeshDecimate.Decimate(sphere, parameters);
            Assert.That(parameters.region is not null && parameters.region != savedRegion );
            Assert.That(decimateResult.facesDeleted > 0);
            Assert.That(decimateResult.vertsDeleted > 0);
        }
    }
}
