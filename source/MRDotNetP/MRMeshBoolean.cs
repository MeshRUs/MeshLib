﻿using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using static MR.DotNet.AffineXf3f;
using static MR.DotNet.MeshBoolean;

namespace MR.DotNet
{
    /// Available CSG operations
    public enum BooleanOperation
    {
        /// Part of mesh `A` that is inside of mesh `B`
        InsideA = 0,
        /// Part of mesh `B` that is inside of mesh `A`
        InsideB = 1,
        /// Part of mesh `A` that is outside of mesh `B`
        OutsideA = 2,
        /// Part of mesh `B` that is outside of mesh `A`
        OutsideB = 3,
        /// Union surface of two meshes (outside parts)
        Union = 4,
        /// Intersection surface of two meshes (inside parts)
        Intersection = 5,
        /// Surface of mesh `B` - surface of mesh `A` (outside `B` - inside `A`)
        DifferenceBA = 6,
        /// Surface of mesh `A` - surface of mesh `B` (outside `A` - inside `B`)
        DifferenceAB = 7,
        Count = 8
    };/// optional parameters for \ref mrBoolean

    public struct BooleanParameters
    {
        public BooleanResultMapper? mapper;
    /// transform from mesh `B` space to mesh `A` space
        public AffineXf3f? rigidB2A;
    /// if set merge all non-intersecting components
        public bool mergeAllNonIntersectingComponents;
    };

    /// output of boolean operation
    public struct BooleanResult
    {
        public Mesh mesh;
    };

    public class MeshBoolean
    {

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRBooleanParameters
        {
            public IntPtr rigidB2A;
            public IntPtr mapper;
            [MarshalAs(UnmanagedType.U1)]
            public bool mergeAllNonIntersectingComponents;
            public IntPtr cb;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRBooleanResult
        {
            public IntPtr mesh;
            public IntPtr errorString;
        }

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRBooleanParameters mrBooleanParametersNew();

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MRBooleanResult mrBoolean( IntPtr meshA, IntPtr meshB, BooleanOperation operation, ref MRBooleanParameters parameters );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrStringData(IntPtr str);

        public static BooleanResult Boolean(Mesh meshA, Mesh meshB, BooleanOperation op )
        {
            return Boolean(meshA, meshB, op, new BooleanParameters());
        }
        public static BooleanResult Boolean(Mesh meshA, Mesh meshB, BooleanOperation op, BooleanParameters parameters )
        {
            MRBooleanParameters mrParameters = mrBooleanParametersNew();
            mrParameters.rigidB2A = parameters.rigidB2A is null ? (IntPtr)null : parameters.rigidB2A.XfAddr();
            mrParameters.mapper = parameters.mapper is null ? (IntPtr)null : parameters.mapper.Mapper;
            mrParameters.mergeAllNonIntersectingComponents = parameters.mergeAllNonIntersectingComponents;
            mrParameters.cb = IntPtr.Zero;

            MRBooleanResult mrResult = mrBoolean(meshA.mesh_, meshB.mesh_, op, ref mrParameters);
            string errorMessage = string.Empty;

            if ( mrResult.errorString != IntPtr.Zero )
            {
                var errData = mrStringData(mrResult.errorString);
                errorMessage = Marshal.PtrToStringAnsi(errData);
            }

            if ( !string.IsNullOrEmpty(errorMessage) )
            {
                throw new SystemException(errorMessage);
            }

            return new BooleanResult
            {
                mesh = new Mesh(mrResult.mesh),
            };
        }
    };
}
