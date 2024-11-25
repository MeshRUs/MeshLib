using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    using static MR.DotNet.ICP;
    using static MR.DotNet.Vector3f;
    public enum ICPMethod
    {
        Combined = 0,     /// PointToPoint for the first 2 iterations, and PointToPlane for the remaining iterations
        PointToPoint = 1, /// select transformation that minimizes mean squared distance between two points in each pair, it is the safest approach but can converge slowly
        PointToPlane = 2  /// select transformation that minimizes mean squared distance between a point and a plane via the other point in each pair,converge much faster than PointToPoint in case of many good (with not all points/normals in one plane) pairs
    };

    /// The group of transformations, each with its own degrees of freedom
    public enum ICPMode
    {
        RigidScale,     /// rigid body transformation with uniform scaling (7 degrees of freedom)
        AnyRigidXf,     /// rigid body transformation (6 degrees of freedom)
        OrthogonalAxis, /// rigid body transformation with rotation except argument axis (5 degrees of freedom)
        FixedAxis,      /// rigid body transformation with rotation around given axis only (4 degrees of freedom)
        TranslationOnly /// only translation (3 degrees of freedom)
    };

    // types of exit conditions in calculation
    public enum ICPExitType
    {
        NotStarted, // calculation is not started yet
        NotFoundSolution, // solution not found in some iteration
        MaxIterations, // iteration limit reached
        MaxBadIterations, // limit of non-improvement iterations in a row reached
        StopMsdReached // stop mean square deviation reached
    };

    /// Stores a pair of points: one samples on the source and the closest to it on the target
    public struct PointPair
    {
        /// coordinates of the source point after transforming in world space
        public Vector3f srcPoint = new Vector3f();

        /// normal in source point after transforming in world space
        public Vector3f srcNorm = new Vector3f();

        /// coordinates of the closest point on target after transforming in world space
        public Vector3f tgtPoint = new Vector3f();

        /// normal in the target point after transforming in world space
        public Vector3f tgtNorm = new Vector3f();

        /// squared distance between source and target points
        public float distSq = 0.0f;

        /// weight of the pair (to prioritize over other pairs)
        public float weight = 1.0f;

        /// id of the source point
        public VertId srcVertId;

        /// for point clouds it is the closest vertex on target,
        /// for meshes it is the closest vertex of the triangle with the closest point on target
        public VertId tgtCloseVert;

        /// cosine between normals in source and target points
        public float normalsAngleCos = 1.0f;

        /// true if if the closest point on target is located on the boundary (only for meshes)
        public bool tgtOnBd = false;

        public PointPair()
        { }

        MRPointPair ToNative()
        {
            MRICPPairData pairData = new MRICPPairData();
            pairData.srcPoint = srcPoint.vec_;
            pairData.srcNorm = srcNorm.vec_;
            pairData.tgtPoint = tgtPoint.vec_;
            pairData.tgtNorm = tgtNorm.vec_;
            pairData.distSq = distSq;
            pairData.weight = weight;

            return new MRPointPair
            {
                ICPPairData = pairData,
                srcVertId = srcVertId,
                tgtCloseVert = tgtCloseVert,
                normalsAngleCos = normalsAngleCos,
                tgtOnBd = tgtOnBd ? (byte)1 : (byte)0
            };
        }
    };

    public struct PointPairs
    {
        public List<PointPair> pairs;
        public BitSet active;
    };

    public struct ICPProperties
    {
        /// The method how to update transformation from point pairs
        public ICPMethod method = ICPMethod.PointToPlane;

        /// Rotation angle during one iteration of PointToPlane will be limited by this value
        public float p2plAngleLimit = (float)System.Math.PI / 6.0f; // [radians]

        /// Scaling during one iteration of PointToPlane will be limited by this value
        public float p2plScaleLimit = 2.0f;

        /// Points pair will be counted only if cosine between surface normals in points is higher
        public float cosThreshold = 0.7f; // in [-1,1]

        /// Points pair will be counted only if squared distance between points is lower than
        public float distThresholdSq = 1.0f; // [distance^2]

        /// Points pair will be counted only if distance between points is lower than
        /// root-mean-square distance times this factor
        public float farDistFactor = 3.0f; // dimensionless

        /// Finds only translation. Rotation part is identity matrix
        public ICPMode icpMode = ICPMode.AnyRigidXf;

        /// If this vector is not zero then rotation is allowed relative to this axis only
        public Vector3f fixedRotationAxis = new Vector3f();

        /// maximum iterations
        public int iterLimit = 10;

        /// maximum iterations without improvements
        public int badIterStopCount = 3;

        /// Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops.
        public float exitVal = 0; // [distance]

        /// a pair of points is formed only if both points in the pair are mutually closest (reciprocity test passed)
        public bool mutualClosest = false;

        public ICPProperties()
        { }

        internal MRICPProperties ToNative()
        {
            return new MRICPProperties
            {
                method = method,
                p2plAngleLimit = p2plAngleLimit,
                p2plScaleLimit = p2plScaleLimit,
                cosThreshold = cosThreshold,
                distThresholdSq = distThresholdSq,
                farDistFactor = farDistFactor,
                icpMode = icpMode,
                fixedRotationAxis = fixedRotationAxis.vec_,
                iterLimit = iterLimit,
                badIterStopCount = badIterStopCount,
                exitVal = exitVal,
                mutualClosest = mutualClosest ? (byte)1 : (byte)0
            };
        }
    };

    public class ICP : IDisposable
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRICPPairData
        {
            public MRVector3f srcPoint;
            public MRVector3f srcNorm;
            public MRVector3f tgtPoint;
            public MRVector3f tgtNorm;
            public float distSq = 0.0f;
            public float weight = 0.0f;
            public MRICPPairData() { }
        };

         [StructLayout(LayoutKind.Sequential)]
        internal struct MRPointPair
        {
            public MRICPPairData ICPPairData;
            public VertId srcVertId;
            public VertId tgtCloseVert;
            public float normalsAngleCos = 0.0f;
            public byte tgtOnBd = 0;

            public MRPointPair() { }
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRICPProperties
        {
            /// The method how to update transformation from point pairs
            public ICPMethod method = ICPMethod.PointToPlane;
            /// Rotation angle during one iteration of PointToPlane will be limited by this value
            public float p2plAngleLimit = (float)System.Math.PI / 6.0f;
            /// Scaling during one iteration of PointToPlane will be limited by this value
            public float p2plScaleLimit = 2.0f;
            /// Points pair will be counted only if cosine between surface normals in points is higher
            public float cosThreshold = 0.7f;
            /// Points pair will be counted only if squared distance between points is lower than
            public float distThresholdSq = 1.0f;
            /// Points pair will be counted only if distance between points is lower than
            /// root-mean-square distance times this factor
            public float farDistFactor = 3.0f;
            /// Finds only translation. Rotation part is identity matrix
            public ICPMode icpMode = ICPMode.AnyRigidXf;
            /// If this vector is not zero then rotation is allowed relative to this axis only
            public MRVector3f fixedRotationAxis;
            /// maximum iterations
            public int iterLimit = 10;
            /// maximum iterations without improvements
            public int badIterStopCount = 3;
            /// Algorithm target root-mean-square distance. As soon as it is reached, the algorithm stops.
            public float exitVal = 0.0f;
            /// a pair of points is formed only if both points in the pair are mutually closest (reciprocity test passed)
            public byte mutualClosest = 0;
            public MRICPProperties() { }
        };

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ref MRICPPairData mrIPointPairsGet(IntPtr pp, ulong idx);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrIPointPairsSize(IntPtr pp);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ref MRICPPairData mrIPointPairsGetRef(IntPtr pp, ulong idx);


        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrICPNew(IntPtr fltObj, IntPtr refObj, float samplingVoxelSize);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrICPNewFromSamples( IntPtr fltObj, IntPtr refObj, IntPtr fltSamples, IntPtr refSamples );

        /// tune algorithm params before run calculateTransformation()
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrICPSetParams(IntPtr icp, ref MRICPProperties prop);

        /// select pairs with origin samples on both objects
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrICPSamplePoints(IntPtr icp, float samplingVoxelSize);

        /// automatically selects initial transformation for the floating object
        /// based on covariance matrices of both floating and reference objects;
        /// applies the transformation to the floating object and returns it
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRAffineXf3f mrICPAutoSelectFloatXf(IntPtr icp);

        /// recompute point pairs after manual change of transformations or parameters
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrICPUpdatePointPairs(IntPtr icp);

        /// returns status info string
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrICPGetStatusInfo(IntPtr icp);

        /// computes the number of samples able to form pairs
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrICPGetNumSamples(IntPtr icp);

        /// computes the number of active point pairs
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern ulong mrICPGetNumActivePairs(IntPtr icp);

        /// computes root-mean-square deviation between points
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern float mrICPGetMeanSqDistToPoint(IntPtr icp);

        /// computes root-mean-square deviation from points to target planes
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern float mrICPGetMeanSqDistToPlane(IntPtr icp);

        /// returns current pairs formed from samples on floating object and projections on reference object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrICPGetFlt2RefPairs(IntPtr icp);

        /// returns current pairs formed from samples on reference object and projections on floating object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrICPGetRef2FltPairs(IntPtr icp);

        /// runs ICP algorithm given input objects, transformations, and parameters;
        /// \return adjusted transformation of the floating object to match reference object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRAffineXf3f mrICPCalculateTransformation(IntPtr icp);

        /// deallocates an ICP object
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern void mrICPFree(IntPtr icp);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrStringData(IntPtr str);

        /// Constructs ICP framework with given sample points on both objects
        /// \param fltObj floating object and transformation from floating object space to global space
        /// \param refObj reference object and transformation from reference object space to global space
        /// \param samplingVoxelSize approximate distance between samples on each of two objects
        public ICP(MeshOrPointsXf fltObj, MeshOrPointsXf refObj, float samplingVoxelSize)
        {
            mrICP_ = mrICPNew(fltObj.mrMeshOrPointsXf_, refObj.mrMeshOrPointsXf_, samplingVoxelSize);
        }
        /// Constructs ICP framework with given sample points on both objects
        /// \param fltObj floating object and transformation from floating object space to global space
        /// \param refObj reference object and transformation from reference object space to global space
        /// \param fltSamples samples on floating object to find projections on the reference object during the algorithm
        /// \param refSamples samples on reference object to find projections on the floating object during the algorithm
        public ICP(MeshOrPointsXf fltObj, MeshOrPointsXf refObj, VertBitSet fltSamples, VertBitSet refSamples)
        {
            mrICP_ = mrICPNewFromSamples(fltObj.mrMeshOrPointsXf_, refObj.mrMeshOrPointsXf_, fltSamples.bs_, refSamples.bs_);
        }

        private bool disposed = false;
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (mrICP_ != IntPtr.Zero)
                {
                    mrICPFree(mrICP_);
                }

                disposed = true;
            }
        }

        ~ICP()
        {
            mrICPFree(mrICP_);
        }

        /// tune algorithm params before run calculateTransformation()
        public void SetParams(ICPProperties prop)
        {
            var icpProp = prop.ToNative();
            mrICPSetParams(mrICP_, ref icpProp);
        }
        /// select pairs with origin samples on both objects
        public void SamplePoints(float sampleVoxelSize)
        {
            mrICPSamplePoints(mrICP_, sampleVoxelSize);
        }
        /// automatically selects initial transformation for the floating object
        /// based on covariance matrices of both floating and reference objects;
        /// applies the transformation to the floating object and returns it
        public void AutoSelectFloatXf()
        {
            mrICPAutoSelectFloatXf(mrICP_);
        }
        /// recompute point pairs after manual change of transformations or parameters
        public void UpdatePointPairs()
        {
            mrICPUpdatePointPairs(mrICP_);
        }
        /// returns status info string
        public string GetStatusInfo()
        {
            var mrStr = mrICPGetStatusInfo(mrICP_);
            return Marshal.PtrToStringAnsi(mrStringData(mrStr));
        }
        /// computes the number of samples able to form pairs
        public int GetNumSamples()
        {
            return (int)mrICPGetNumSamples(mrICP_);
        }
        /// computes the number of active point pairs
        public int GetNumActivePairs()
        {
            return (int)mrICPGetNumActivePairs(mrICP_);
        }
        /// computes root-mean-square deviation between points
        public float GetMeanSqDistToPoint()
        {
            return mrICPGetMeanSqDistToPoint(mrICP_);
        }
        /// computes root-mean-square deviation from points to target planes
        public float GetMeanSqDistToPlane()
        {
            return mrICPGetMeanSqDistToPlane(mrICP_);
        }
        /// returns current pairs formed from samples on floating object and projections on reference object
        public PointPairs GetFlt2RefPairs()
        {
            var mrPairs = mrICPGetFlt2RefPairs(mrICP_);
            int numPairs = (int)mrIPointPairsSize(mrPairs);

            PointPairs res = new PointPairs();
            res.pairs = new List<PointPair>(numPairs);

            for ( int i = 0; i < numPairs; ++i )
            {
                var mrPair = mrIPointPairsGet(mrPairs, (ulong)i);

                PointPair pair = new PointPair();
                pair.srcPoint = new Vector3f(mrPair.srcPoint);
                pair.srcNorm = new Vector3f(mrPair.srcNorm);
                pair.tgtPoint = new Vector3f(mrPair.tgtPoint);
                pair.tgtNorm = new Vector3f(mrPair.tgtNorm);
                pair.weight = mrPair.weight;
                pair.distSq = mrPair.distSq;                

                res.pairs.Add( pair );
            }

            return res;
        }
        /// returns current pairs formed from samples on reference object and projections on floating object
        public PointPairs GetRef2FltPairs()
        {
            var mrPairs = mrICPGetRef2FltPairs(mrICP_);
            int numPairs = (int)mrIPointPairsSize(mrPairs);

            PointPairs res = new PointPairs();
            res.pairs = new List<PointPair>(numPairs);

            for (int i = 0; i < numPairs; ++i)
            {
                var mrPair = mrIPointPairsGet(mrPairs, (ulong)i);

                PointPair pair = new PointPair();
                pair.srcPoint = new Vector3f(mrPair.srcPoint);
                pair.srcNorm = new Vector3f(mrPair.srcNorm);
                pair.tgtPoint = new Vector3f(mrPair.tgtPoint);
                pair.tgtNorm = new Vector3f(mrPair.tgtNorm);
                pair.weight = mrPair.weight;
                pair.distSq = mrPair.distSq;

                res.pairs.Add(pair);
            }

            return res;
        }
        /// runs ICP algorithm given input objects, transformations, and parameters;
        /// \return adjusted transformation of the floating object to match reference object
        public AffineXf3f CalculateTransformation()
        {
            var mrXf = mrICPCalculateTransformation(mrICP_);
            return new AffineXf3f(mrXf);
        }

        internal IntPtr mrICP_;

    }
}
